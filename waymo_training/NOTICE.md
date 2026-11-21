# Waymo dataset and model terms

The Waymo-based experiments in this directory use the **Waymo Open Dataset End-to-End Driving v1.0.0**, provided by Waymo LLC under the [Waymo Dataset License Agreement for Non-Commercial Use](https://waymo.com/open/terms/). Use of the dataset and its trained models is subject to the agreement. Section 4 restricts use in operating or assisting a vehicle, in production systems, and for primarily commercial purposes.

Use the following Waymo-provided attribution from §1 in Waymo-derived work, including model and research artifacts.

```bibtex
@misc{waymo_open_dataset,
  title = {Waymo Open Dataset: An autonomous driving dataset},
  website = {\url{https://www.waymo.com/open}},
  year = {2019-2025}
}
```

Keep downloaded records and extracted features in private runtime storage. The agreement places conditions on redistribution of the dataset and its modifications. Any permitted release of trained architectures, weights, biases, or associated derivatives must include this attribution, a copy of the agreement, and a prominent notice that the full release and downstream use or modification are governed by the agreement, including its non-commercial restrictions. See §§1, 2, and 4. [Waymo's agreement](https://waymo.com/open/terms/) · [Waymo's FAQ](https://waymo.com/open/faq/)

The simulator's [MIT license](../LICENSE) covers the legacy simulator software. The Waymo agreement governs the Waymo data and the data-derived model artifacts described here. Any reused upstream schema files carry their own license notices.
