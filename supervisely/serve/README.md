<div align="center" markdown>

<img src=""/>  

# Serve Metric Learning

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
</p>

[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/gl-metric-learning/supervisely/serve)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/gl-metric-learning/supervisely/serve&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/gl-metric-learning/supervisely/serve&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/gl-metric-learning/supervisely/serve&counter=runs&label=runs&123)](https://supervise.ly)

</div>

# Overview

Serve Metric Learning model as Supervisely Application.

Application key points:
- Deployed on GPU or CPU
- Can be used with Supervisely Applications or [API](https://github.com/supervisely-ecosystem/gl-metric-learning/blob/main/supervisely/serve/src/demo_api_requests.py)
- Choose from 5 models, pretrained on different domains 


# How to Run

1. Add [Serve Metric Learning](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Ffairmot%252Fsupervisely%252Fserve) from ecosystem to your team  

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/gl-metric-learning/supervisely/serve" src="https://i.imgur.com/IQvX8TG.png" width="350px" style='padding-bottom: 10px'/>

2. Choose model, deploying device and press the **Run** button

<img src="https://i.imgur.com/RCkK10C.png" width="80%" style='padding-top: 10px'>  

3. Wait for the model to deploy
<img src="https://i.imgur.com/xbt4Aqj.png" width="80%">  


# Acknowledgment

This app is based on the great work `Reviving Iterative Training with Mask Guidance for Interactive Segmentation` ([paper](https://arxiv.org/abs/2010.01650),  [github](https://github.com/psinger/kaggle-landmark-recognition-2020-1st-place)). ![GitHub Org's stars](https://img.shields.io/github/stars/psinger/kaggle-landmark-recognition-2020-1st-place?style=social)
