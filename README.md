# Overview

nanovvpo is a minimal codebase to reproduce RL algorithms for LLMs such as GRPO, VPPO and RFT.

This is a training curve obtained by `scripts/grpo_cd_3b.sh` and generated with `miniplot.py` for GRPO training on Countdown 3to4 (from Jiayi) for Qwen-2.5-3B-Instruct.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/4fb2ff8f-ab31-4299-b8f5-ce470ab80e41" />

## What Can It Do

nanovppo was developed as an educational tool for experiments with RL for LLMs. It represents a self-contained, dependency-lean codebase that reimplements existing well-known algorithms such as: 

GRPO: https://arxiv.org/abs/2501.12948 

VPPO: https://arxiv.org/pdf/2410.01679 

RFT: https://arxiv.org/abs/2203.14465 

nanovvpo relies on SGLang (https://docs.sglang.ai/start/install.html), and VLLM (https://github.com/vllm-project/vllm) to speed-up model rollouts that are required to make the algorithms efficient. It also supports multi-gpu training in the same node and could be easily extended to multiple nodes. 

## Intended Uses

nanovppo is best suited for practitioners that want to ramp up quickly on these algorithms for educational purposes. 

nanovppo is being shared with the research community to foster further research in this area. 

## Out of Scope Uses

nanovppo is not well suited for training large scale models.

## Get Started

To begin using nanovppo, just look at scripts/*.sh in the codebase for an example on how to train a model. You will need at least 2 gpus (one for inference, one for training).

## Evaluation

As a proof of concept use case, we provide code to finetune models with GRPO, RFT and VPPO on the MATH dataset (https://github.com/hendrycks/math). 

## Limitations

nanovppo was developed for research and experimental purposes. Further testing and validation are needed before considering its application in commercial or real-world scenarios. 

nanovppo was designed and tested using the English language. Performance in other languages may vary and should be assessed by someone who is both an expert in the expected outputs and a native speaker of that language. 

nanovppo should not be used in highly regulated domains where inaccurate outputs could suggest actions that lead to injury or negatively impact an individual's legal, financial, or life opportunities. 

nanovppo inherits any biases, errors, or omissions produced by its base model. Developers are advised to choose an appropriate base model carefully, depending on the intended use case. 

nanovppo inherits any biases, errors, or omissions characteristic of its training data, which may be amplified by any AI-generated interpretations.  

## Best Practices

Users are responsible for sourcing their datasets legally and ethically. This could include securing appropriate copy rights, ensuring consent for use of audio/images, and/or the anonymization of data prior to use in research.    

Users are reminded to be mindful of data privacy concerns and are encouraged to review the privacy policies associated with any models and data storage solutions interfacing with nanovppo.  

It is the user’s responsibility to ensure that the use of nanovppo complies with relevant data protection regulations and organizational guidelines. 

## Contact

We welcome feedback and collaboration from our audience. If you have suggestions, questions, or observe unexpected/offensive behavior in our technology, please contact us at alsordon@microsoft.com. 

If the team receives reports of undesired behavior or identifies issues independently, we will update this repository with appropriate mitigations. 

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
