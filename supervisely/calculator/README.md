<div align="center" markdown>
<img src=""/>


# Import CSV catalog

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Use">How To Use</a> •
  <a href="#Results">Results</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/import-csv-catalog)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/import-csv-catalog)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/import-csv-catalog&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/import-csv-catalog&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/import-csv-catalog&counter=runs&label=runs&123)](https://supervise.ly)

</div>

# Overview

Application creates Supervisely images project from csv catalog

Application key points:  
- Name of the `csv` file determines created project name
- Each product from the catalog will be assigned to class `product` with appropriate product id(number) tag
- Product info will be stored in `object properties` -> `Data`
- Images names will be taken from Image URL
- Required fields: **Image URL**, **PRODUCT ID**

Example [CSV catalog](https://github.com/supervisely-ecosystem/import-csv-catalog/releases/download/v0.0.1/test_snacks_catalog.csv)

# How to Use
1. Add [Import CSV catalog](https://ecosystem.supervise.ly/apps/import-csv-catalog) to your team from Ecosystem.

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/import-csv-catalog" src="https://i.imgur.com/akcxVrR.png" width="350px" style='padding-bottom: 20px'/>  

2. Run app from the context menu of the `csv` file:

<img src="https://i.imgur.com/8s9IREE.png" width="100%"/>

# Results

After running the application, you will be redirected to the `Tasks` page. Once application processing has finished, your project will be available. Click on the project name to proceed to it.

<img src="https://i.imgur.com/G4BOKH4.png"/>
