<div align="center">
<a href="https://blackhillsinfosec.com"><img width="100%" src="https://github.com/her3ticAVI/BlabberFish/blob/main/images/BlabberFishLogo.png" alt="BlabberFish Logo" /></a>
<hr>

  <a href="https://github.com/blackhillsinfosec/WifiForge/actions"><img alt="GitHub Workflow Status" src="https://img.shields.io/github/actions/workflow/status/her3ticAVI/BlabberFish/.github%2Fworkflows%2Fpython-app.yml?style=flat-square"></a> 
  &nbsp;
  <a href="https://discord.com/invite/bhis"><img alt="Discord" src="https://img.shields.io/discord/967097582721572934?label=Discord&color=7289da&style=flat-square" /></a>
  &nbsp;
  <a href="https://github.com/blackhillsinfosec/WifiForge/graphs/contributors"><img alt="npm" src="https://img.shields.io/github/contributors-anon/her3ticAVI/BlabberFish?color=yellow&style=flat-square" /></a>
  &nbsp;
  <a href="https://x.com/BHinfoSecurity"><img src="https://img.shields.io/badge/follow-BHIS-1DA1F2?logo=twitter&style=flat-square" alt="BHIS Twitter" /></a>
  &nbsp;
  <a href="https://x.com/BHinfoSecurity"><img src="https://img.shields.io/github/stars/her3ticAVI/BlabberFish?style=flat-square&color=rgb(255%2C218%2C185)" alt="BlabberFish Stars" /></a>
  
<p class="align center">
<h4><code>BlabberFish</code> is a dedicated web server application engineered to simplify and automate the documentation of audio conversations. It takes raw audio files and converts them into structured, readable text transcripts.</h4>
</p>

<div style="text-align: center;">
  <h4>
    <a target="_blank" href="https://blabberfish.github.io/" rel="dofollow"><strong>Explore the Docs</strong></a>&nbsp;·&nbsp;
    <a target="_blank" href="https://discord.com/invite/bhis" rel="dofollow"><strong>Community Help</strong></a>&nbsp;·&nbsp;
    <a target="_blank" href="https://www.blackhillsinfosec.com/blabberfish/" rel="dofollow"><strong>Blog Post</strong></a>
  </h4>
</div>
<hr>
<a href="https://blackhillsinfosec.com"><img width="75%" height="75%" src="https://github.com/her3ticAVI/BlabberFish/blob/main/images/BlabberFish-running.png" alt="BlabberFish Running" /></a>
<div align="left">

## Navigation

- [Installation Documentation](https://blabberfish.github.io/Installation)
- [Walkthroughs](https://blabberfish.github.io/Lab-Walkthroughs/)
- [Troubleshooting](https://blabberfish.github.io/Troubleshooting)
- [Overview](https://blabberfish.github.io/Overview)
- [Contributing](https://blabberfish.github.io/Development)

## Example Usage

```shell
{{ TODO }}

```

## Setup

### Installation

The following command should set up the environment required to use BlabberFish. Only Linux is supported at this time.

```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda env create -f environment.yml
conda activate BabbelFish-venv
sudo apt install ffmpeg
```
### Hugging Face Token

For the `transcribe_pyannote.py` script, `pyannote.audio` diarization requires access to pretrained models on HuggingFace.

Get a free token at `https://huggingface.co/settings/tokens` and then make it read only access, then search for `pyannote/speaker-diarization` model and accept the license.

## References

- https://github.com/mr-pmillz/transcribe

<div align="center">

Made with ❤️  by Black Hills Infosec
