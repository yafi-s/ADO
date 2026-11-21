# Rider Follow on the Urban Pace drive

Town03 seed 212 produced 3,354 matched exposures, CARLA frames 2212 through 5565 at 10 Hz. Across eight major turns, body yaw reached **26.70°/s**. The recorded heading crossed ±180° at frames 3915 and 4843. Every simulator frame matched its camera frame, video index and native camera timestamp within 1 ms. The raw replay digest matched the capture manifest. [Per-turn measurements and source fingerprint](follow-turns.json)

Replaying these headings at a modeled 60 Hz, the former camera trailed body yaw by 333 ms and had a 95th-percentile error of **8.390°** with movie footage visible. The updated camera measured zero lag at 16.7 ms resolution and **0.618°** error. Its worst error was 1.042°. The earlier seed 211 recording measured 0.597° at the same percentile with the updated camera. [Earlier recording and method](../perception/rider-turns/README.md)

At native exposure boundaries, peak angular jerk fell from 45.687 to 25.923°/s³ and RMS jerk fell from 6.640 to 6.088°/s³. The 95th percentile rose slightly from 13.174 to 13.751°/s³. With camera footage hidden and the whole rider scene interpolated at 60 Hz, the 95th percentile fell from 59.137 to 14.008°/s³. The comparison applies both camera algorithms to the same vehicle trajectory.
