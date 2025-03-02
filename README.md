# <b>HYDRO: Towards Non-Reversible Face De-Identification Using a High-Fidelity Hybrid Diffusion and Target-Oriented Approach</b>

## Abstract
>Target-oriented face de-identification models aim to anonymize facial identities by transforming original faces to resemble specific targets. 
>Such models commonly leverage generative encoder-decoder architectures to manipulate facial appearances, enabling them to produce realistic high-fidelity de-identification results,
>while ensuring considerable attribute-retention capabilities. However, target-oriented models also carry the risk of inadvertently preserving subtle identity cues,
>making them (potentially) reversible and susceptible to reconstruction attacks. To address this problem, we introduce a novel robust face de-identification approach,
>called HYDRO, that combines target-oriented models with a dedicated diffusion process specifically designed to destroy any imperceptible information that may allow learning to reverse the de-identification procedure.
>HYDRO first de-identifies the given face image, injects noise into the de-identification result to impede reconstruction,
>and then applies a diffusion-based recovery step to improve fidelity and minimize the impact of the noising process on the data characteristics.
>To further improve image fidelity and better retain gaze directions, a novel Eye Similarity Discriminator (ESD) is also introduced and incorporated it into the training of HYDRO.
>Extensive quantitative and qualitative experiments on three datasets demonstrate that HYDRO exhibits state-of-the-art (SOTA) fidelity and attribute-retention capabilities,
>while being the only target-oriented method resilient against reconstruction attacks. In comparison to multiple SOTA competitors, HYDRO substantially reduces the success of reconstruction attacks by 85.7% on average.

## Getting Started
Coming soon...

## License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png"/></a>
