"""Fit a small action-value tree to paired out-and-back vs stay-in-lane traffic rollouts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .stationary_intent import FEATURES, MODEL, VERSION, StationaryIntent


def generate(seed, count, *, stress=False):
    """One independent encounter per row. No route, fixture data or user collision labels."""
    rng = np.random.default_rng(seed)
    n = count
    gap = rng.uniform(5., 46., n)
    ego = np.minimum(rng.uniform(0., 14., n), np.sqrt(2.8 * np.maximum(0., gap - 4.)))
    settled = rng.random(n) < .30
    gap[settled] = rng.uniform(5., 12., settled.sum())
    ego[settled] = rng.uniform(0., .28, settled.sum())
    observe = rng.uniform(.25, 11., n)
    waiting = np.zeros(n)
    waiting[settled] = rng.uniform(0., 38., settled.sum())
    observe += waiting
    flow = rng.uniform(7., 12., n)
    accel = rng.uniform(.65, 2.4, n)
    mode = rng.choice(4, size=n, p=[.55,.13,.21,.11] if not stress else [.63,.08,.17,.12])
    # Temporary queued traffic, long stationary obstruction, ordinary slow
    # traffic and a vehicle already departing. These hidden modes and future
    # release times are available solely to the synthetic traffic engine.
    dwell = rng.lognormal(np.log(3.2 if not stress else 5.), .55 if not stress else .76, n)
    arrival = np.where(settled,0.,(gap-rng.uniform(18.,28.,n)) / np.maximum(3., ego))
    # The residual stop clock can be uncorrelated with us or begin when
    # traffic closes behind the vehicle. Mixed timing keeps long observation
    # from being equivalent to a genuine stopped wait near the lead.
    independent = rng.random(n) < .38
    release = np.where(independent, dwell - observe, dwell + arrival - waiting)
    release = np.where(mode==1, rng.uniform(45.,100.,n)-waiting,release)
    release = np.where(mode==2, -rng.uniform(7.,40.,n),release)
    release = np.where(mode==3, -rng.uniform(0.,4.,n),release)
    terminal = np.where(mode==2,rng.uniform(.75,3.3,n),rng.uniform(5.5,12.,n))
    def lead_speed(t):
        return np.clip((t-release)*accel,0.,terminal)
    v0 = lead_speed(0.)
    # Matching and motion estimation are noisy, especially on a newly
    # observed surface. All inference features come from t <= 0.
    previous = np.clip(-observe, -.85, 0.)
    previous_mid = np.minimum(previous*.82,-.001)
    current_mid = previous*.22
    noise = .09 + .09 / np.sqrt(np.maximum(observe,.25))
    old = np.maximum(0.,lead_speed(previous_mid)+rng.normal(0.,noise))
    now = np.maximum(0.,lead_speed(current_mid)+rng.normal(0.,noise))
    derivative = np.clip((now-old)/np.maximum(.22,current_mid-previous_mid),-5.,5.)
    actual_wait = np.where((ego<.4)&(gap<15.)&(v0<.35),waiting,0.)
    front_v = rng.uniform(0.,13.,n)
    front = rng.uniform(8.,58.,n)
    free = rng.random(n)<.70
    projected_target = np.where(free,60.,front+4.*front_v)
    gain = projected_target-(gap+4.*now)
    x = np.column_stack((gap,ego,now,old,derivative,actual_wait,observe,gain))
    # Both choices see exactly the same future lead and adjacent-lane traffic.
    # The utilities compare actual distance, time occupying another lane and
    # the cost of an unfinished pass. A fast departing lead can outrun us.
    dt=.25
    horizon=12.
    source_pos=gap-2.45
    adjacent_pos=np.where(free,250.,front-2.45)
    stay_pos=np.zeros(n)
    stay_v=ego.copy()
    outbound=rng.uniform(2.,3.4,n)
    inbound=rng.uniform(2.2,3.8,n)
    lane_cap=np.minimum(flow,rng.uniform(5.5,9.,n))
    # Waiting is a reversible action. Compare immediate commitment to both
    # staying and deferring until the lead has had time to show new motion.
    # The delayed policy uses only the lead velocity at the later decision.
    starts=(0.,3.5)
    trajectories=[dict(position=np.zeros(n),v=ego.copy(),finished=np.full(n,np.inf),
         elapsed=np.zeros(n),attempted=np.zeros(n,bool),unsafe=np.zeros(n)) for _ in starts]
    def move(v,position,limit,v_lead,gap_now):
        desired=np.minimum(limit,v_lead+np.maximum(0.,gap_now-4.)/1.7)
        a=np.clip(1.1*(desired-v),-4.6,2.1)
        next_v=np.maximum(0.,v+a*dt)
        return next_v,position+.5*(v+next_v)*dt
    for step in range(int(horizon/dt)):
        t=step*dt
        lv=lead_speed(t)
        stay_v,stay_pos=move(stay_v,stay_pos,flow,lv,source_pos-stay_pos)
        for start,action in zip(starts,trajectories):
            pos,v=action["position"],action["v"]
            if t==start:
                action["attempted"][:]=True if start==0. else lv<.55*flow
            attempt=action["attempted"]
            front_gap=pos-source_pos-5.
            required=3.+1.6*lv
            start_return=attempt & (action["finished"]==np.inf)&(t>=start+outbound)&(front_gap>=required)
            action["finished"][start_return]=t+inbound[start_return]
            across=(t<start+outbound)|(t>action["finished"]-inbound)
            active=attempt&(t<action["finished"])
            action["elapsed"]+=active*dt
            # A rejoining ego can force the source vehicle to consume its
            # response headway as it accelerates. Score that deficit explicitly.
            returning=attempt&(t>=action["finished"]-inbound)
            action["unsafe"]=np.maximum(action["unsafe"],np.where(returning,required-front_gap,0.))
            own=(~attempt)|(t<start+.6*outbound)
            ahead_v=np.where(own,lv,front_v)
            ahead=np.where(own,source_pos,adjacent_pos)
            action["v"],action["position"]=move(v,pos,np.where(across&active,lane_cap,flow),ahead_v,ahead-pos)
        source_pos+=.5*(lv+lead_speed(t+dt))*dt
        adjacent_pos+=front_v*dt
    def utility(start,action):
        attempt=action["attempted"]
        # Explicit instructor preference: give a stopped motor vehicle the
        # chance to resume in order. If it restarts during this pass, the
        # alternative must rejoin behind it and pays the same abort costs.
        # Future release and hidden simulation timing are labels only.
        restart_during_pass=(release>=start)&(release<=np.minimum(horizon,action["finished"]))
        incomplete=attempt&((action["finished"]>horizon)|restart_during_pass)
        # An uncompleted pass cannot claim terminal distance ahead in the
        # adjacent lane: the modeled route must remain reachable. Give its
        # abort behind the source the distance the follower could have made.
        useful=np.where(incomplete,np.minimum(action["position"],stay_pos),action["position"])
        return useful-3.5*attempt-.4*action["elapsed"]-12.*incomplete-3.*action["unsafe"]
    immediate,delayed=[utility(start,action) for start,action in zip(starts,trajectories)]
    gain_pass=immediate-np.maximum(stay_pos,delayed)
    return x,gain_pass,mode,release,immediate-delayed,immediate-stay_pos


def tree_fit(x,y):
    nodes=[]
    def build(indices, depth):
        k=len(nodes);nodes.append([-1,0.,-1,-1])
        if depth>=7 or len(indices)<300:
            return k
        yi=y[indices];best=None
        for j,column in enumerate(x[indices].T):
            order=np.argsort(column,kind="stable");values=column[order]
            cuts=np.flatnonzero(np.diff(values)>1e-6)+1
            cuts=cuts[(cuts>=130)&(cuts<=len(values)-130)]
            if not len(cuts):
                continue
            cumulative=np.cumsum(yi[order]);a=cumulative[cuts-1];total=cumulative[-1]
            loss=a*a/cuts+(total-a)**2/(len(values)-cuts)
            i=int(np.argmax(loss));cut=int(cuts[i])
            if best is None or loss[i]>best[0]:
                best=(loss[i],j,(values[cut-1]+values[cut])*.5,order,cut)
        if best is not None:
            _,j,threshold,order,cut=best
            nodes[k]=[j,threshold,build(indices[order[:cut]],depth+1),build(indices[order[cut:]],depth+1)]
        return k
    build(np.arange(len(x)),0)
    return np.asarray(nodes)


def leaves(tree,x):
    node=np.zeros(len(x),dtype=int)
    branch=tree[node,0]>=0.
    while branch.any():
        n=node[branch];j=tree[n,0].astype(int)
        node[branch]=np.where(x[branch,j]<=tree[n,1],tree[n,2],tree[n,3]).astype(int)
        branch=tree[node,0]>=0.
    return node


def evaluate(x,y,mode,tree,gains,prob):
    n=leaves(tree,x)
    passing=gains[n]>0.
    regret=np.maximum(0.,y)-np.where(passing,y,0.)
    def stats(mask):
        if not mask.any():return {}
        return dict(rows=int(mask.sum()),defer_fraction=round(float(passing[mask].mean()),4),
                    oracle_choice_accuracy=round(float(((y[mask]>0.)==passing[mask]).mean()),4),
                    mean_regret_m=round(float(regret[mask].mean()),3))
    return dict(all=stats(np.ones(len(x),bool)),
                baseline_always_defer_regret_m=round(float(np.mean(np.maximum(0.,-y))),3),
                baseline_always_follow_regret_m=round(float(np.mean(np.maximum(0.,y))),3),
                brier=round(float(np.mean((prob[n]-(y<=0.))**2)),4),
                newly_observed_stationary=stats((x[:,2]<.3)&(x[:,3]<.3)&(x[:,5]<1.)),
                extended_wait=stats((x[:,2]<.3)&(x[:,5]>18.)),
                sustained_slow=stats((mode==2)&(x[:,2]>.7)&(x[:,2]<3.3)),
                departing=stats((x[:,2]>1.)&(x[:,4]>.8)))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output",type=Path,default=MODEL)
    args=cli.parse_args()
    x,y,*_=generate(41387,50000)
    tree=tree_fit(x,y)
    cx,cy,*_=generate(219109,30000)
    cn=leaves(tree,cx)
    gains=np.zeros(len(tree));prob=np.zeros(len(tree));counts=[]
    for n in np.flatnonzero(tree[:,0]<0):
        labels=cy[cn==n]
        if len(labels)<20:
            raise RuntimeError("Too little independent leaf calibration")
        gains[n]=labels.mean();prob[n]=(labels<=0.).mean();counts.append(len(labels))
    output=args.output;output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,version=VERSION,features=FEATURES,feature=tree[:,0].astype(int),threshold=tree[:,1],
        left=tree[:,2].astype(int),right=tree[:,3].astype(int),pass_gain_m=gains,follow_fraction=prob)
    check=StationaryIntent(output);check.predict(dict(zip(FEATURES,x[0])))
    report=dict(schema=VERSION,checkpoint_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        fit_rows=len(x),fit_seed=41387,calibration_rows=len(cx),calibration_seed=219109,leaves=len(counts),
        minimum_calibration_support=min(counts),
        scope="Synthetic longitudinal counterfactual action values conditional on sensor-confirmed source motor vehicle. No native tape, route, staged actor, fixture seed, runtime label or target incident enters the fit. The separate geometric planner owns permission.",
        actions=["follow","defer_to_existing_choice"],features=list(FEATURES),decision="Defer if empirical mean counterfactual pass gain exceeds zero; otherwise stay in source lane.",
        probability="Independent calibration-leaf fraction of rollouts with lower/equal cost for following. A cost-sensitive decision need not have a majority probability.",
        assumptions=dict(horizon_s=12.,time_step_s=.25,pass_effort_m=3.5,other_lane_cost_m_per_s=.4,unfinished_pass_m=12.,
            hidden_modes=["temporary_stop","long_stop","sustained_slow","departing"],
            default_mode_probabilities=[.55,.13,.21,.11],default_median_short_dwell_s=3.2),
        input_source_sha256={f"crosswalk/{p.name}":hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__),Path(__file__).with_name("stationary_intent.py"))},held_out={})
    for seed,stress in ((718813,False),(983239,True)):
        a,b,h,*_=generate(seed,16000,stress=stress)
        report["held_out"][str(seed)]=dict(stress=stress,**evaluate(a,b,h,tree,gains,prob))
    report["assumptions"].update(
        provenance="Traffic-mode weights, mixture fractions, driver preferences and speed distributions are simulation design assumptions; they are not measured population frequencies. Retained native instructor seed 75 exhibits a stop followed by restart, and the repository journey fixture schedules short holds. Neither its telemetry nor any native capture is a fit input.",
        temporary_hold_distributions="Default lognormal median 3.2s sigma 0.55. Stress median 5s sigma 0.76. 38% have a clock independent of our approach; the rest vary their residual wait with queue arrival.",
        stress_mode_probabilities=[.63,.08,.17,.12],
        measured_alignment="Ego speed, source gap/speed, earlier speed and accel, stationary close wait, association age and projected lane gain are all causal. A separate camera/lidar/radar typed identity gate is required before querying the head.",
        counterfactual="Exit at bounded speed, rejoin only with physical lead clearance + speed-dependent headway, pay for unfinished exposure and post-return headway deficit. A source motor vehicle restarting before completion is treated as a queue-priority abort; the maximum retained progress is the source-following position. This is a chosen behavioral preference, not a crash measurement.",
        alternatives="Follow labels have the hindsight upper bound of two actual simulated alternatives: stay for the horizon, or wait 3.5s and reevaluate using the source velocity then. Report alternate fixed-policy regret separately.",
        continued_front_bound="Passing simulations conservatively keep the tracked target-front bound even after our return during the finite horizon.",
        regret_units="Equivalent metres, including 3x post-return headway deficit in metres and fixed effort, exposure and abort terms.")
    # The model and development objective were frozen before revealing this
    # third evaluation seed. The earlier two evaluations served development.
    report["development_evaluations"]=report.pop("held_out")
    seed=62739539
    a,b,h,_,delay_only,stay_only=generate(seed,32000)
    report["untouched_final_evaluation"]=dict(seed=seed,rows=len(a),**evaluate(a,b,h,tree,gains,prob))
    choice=gains[leaves(tree,a)]>0.
    report["fixed_alternative_sensitivity"]={
        name:dict(mean_regret_m=float(np.mean(np.maximum(0.,adv)-np.where(choice,adv,0.))),
                  always_follow_regret_m=float(np.mean(np.maximum(0.,adv))),
                  always_defer_regret_m=float(np.mean(np.maximum(0.,-adv))))
        for name,adv in (("delay_then_causal_reevaluation",delay_only),("stay_entire_horizon",stay_only))}
    examples={"fresh_stationary_far":[35.,8.,.08,.05,.05,0.,1.,30.],
              "short_wait_near":[7.,.1,0.,.1,-.1,2.,9.,30.],
              "persistent_stationary_near":[7.,.1,0.,.1,-.1,22.,29.,30.],
              "sustained_slow":[18.,6.,1.7,1.7,0.,0.,4.,30.],
              "source_accelerating":[18.,6.,2.,.8,2.,0.,2.,30.]}
    report["illustrative_feature_probes"]={}
    for name,row in examples.items():
        decision,probability=check.predict(row)
        leaf=leaves(tree,np.array(row).reshape(1,-1))[0]
        report["illustrative_feature_probes"][name]=dict(inputs=row,decision=decision,
            follow_fraction=probability,estimated_pass_gain_equivalent_m=float(gains[leaf]))
    report["rejected_objective"]="The initial terminal-distance-only comparison repeatedly recommended passing a new stationary car. Aggregate synthetic accuracy did not address the instructor preference for queued traffic. It was not promoted."
    path=output.with_name("stationary-intent-fit.json")
    path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report["untouched_final_evaluation"]))

if __name__=="__main__":
    main()
