# AURA Dataset: Cervical Mucus Images (v0.2)
The AURA Dataset is a living, exploratory image dataset built in a bachelor-thesis context to understand real-world cervical mucus uploads and to support image usability filtering. The current documented snapshot is version `0.2` with an authoritative total of `816` images based on ground-truth records.

## Summary
### Dataset Identity
#### Name
AURA Dataset: Cervical Mucus Images

#### Acronym
AURA (`Ausflussueberwachung und regelmaessige Analyse`)

#### Version
`v0.2` (living dataset, not frozen)

#### Recommended Citation
Schueler, Emma (2026). AURA Dataset: Cervical Mucus Images, v0.2 (private research dataset, unpublished).

### Intended Scope
#### Primary Intended Use
Differentiate images that are usable for further analysis from images that are not usable, with strong focus on blur and setup validity.

#### Explicitly Out of Scope
- Cycle phase inference.
- Fertility state prediction.
- Diagnosis or clinical decision support.

### Dataset Size and Sources
#### Authoritative Total
`816` images (ground-truth total).

#### Source Breakdown
- App volunteers: `496`
- Web: `172`
- Professional-shared client images: `132`
- FemTech company survey images: `16`

## Authorship and Governance
### Responsible Parties
#### Creator
Emma Schueler

#### Affiliation Context
Work conducted in a bachelor-thesis context at HTW Berlin; this dataset is not documented as an official institutional release or endorsement.

#### Owner / Data Steward
Emma Schueler

#### Contact
`hello@dischargediary.com`

### Access and Distribution
#### Current Access
Raw images are locally accessible only to the creator.

#### Public Availability
Not public as of February 9, 2026. External sharing and redistribution are currently blocked because consent/legal coverage is incomplete across sources.

## Collection and Provenance
### Collection Window
#### Active Collection Period
December 1, 2025 to February 9, 2026.

#### Timestamp Caveat
Some files have timestamps from before December 1, 2025 because users could upload pre-existing images from local device storage.

### Source-Specific Provenance
#### App Volunteers
Collected through a custom Progressive Web App (PWA). Uploads can be traced to pseudonymous user IDs/tokens, and consent can be matched via email-based workflows.

#### Web
Downloaded from publicly accessible internet pages; per-image license metadata is generally incomplete. Public accessibility is not treated as blanket reuse or redistribution permission.

#### Professional-Shared Client Images
Shared through a practitioner-mediated process; consent evidence is partly informal and source-level rather than complete, centrally archived per-image legal documentation.

#### FemTech Company Survey Images
Provided directly by the company for thesis and model-training use; publication and redistribution rights are not yet documented as granted in this version.

### Capture Modality
#### Device and Platform Constraints
No device restrictions were enforced. Any user able to access the PWA in a browser could upload images (including iOS and Android devices), from camera capture or existing media.

## Data Content
### Population and Context
#### App Participant Profile (Reported)
- Approximate age range: `18-55`
- Geography: primarily Germany

#### Metadata Reliability
User-entered metadata (for example cycle details or contraceptive context) is currently treated as unverified; image content is treated as the reliable component.

### Upload Guidance Given to Users
#### Instruction Text
- "Perfect shots I need you to upload: Clear, on toilet paper or fingers."
- "Bad shots I also need you to upload: Shake the camera, turn off the lights, or take a blurry photo from far away."

### File Types and Size
#### File Formats (Folder Snapshot)
- jpg: `379`
- jpeg: `303`
- png: `112`
- webp: `23`
- heic: `2`
- svg: `1`
- no extension: `3`

#### File Size Range (Folder Snapshot)
- Minimum: `4,552` bytes (~4.45 KB) at `Images/Web52/IMG_3906 4.jpg`
- Maximum: `5,940,294` bytes (~5.94 MB) at `Images/AURA101/5b5144cc-4fc9-4164-8d27-197fd7b6c089.jpg`

#### Snapshot Caveat
Folder snapshots and format counts may not match the authoritative ground-truth total due to later file deletions/moves.

## Annotation Schema
### Annotation Process
#### Annotators
Initial labeling was performed by Emma Schueler; an expert reviewer provided feedback for uncertain cases.

#### Annotation Style
Multi-label annotation is used. Label counts represent occurrences and can exceed the number of unique images.

### Label Sets and Counts
#### Quality (target set size: 816 images)
- Usable: `551`
- Too Blurry: `136`
- Wrong Setup: `121`
- Irrelevant: `10`

#### Types (subset size: 385 images)
- underwear: `104`
- finger: `200`
- tissue: `67`
- other: `14`

#### Blur (target set size: 816 images)
- no blur: `264`
- blurry: `72`
- blurry background: `415`
- blurry foreground: `65`

#### Usability Considering Blur (target set size: 816 images)
- focused enough: `671`
- too blurry: `145`

#### Usability Considering NFP (target set size: 816 images)
- usable: `560`
- not usable: `256`
- unsure: `0`

### Operational Label Definitions
#### focused enough vs too blurry
The threshold is conservative: only images with almost no blur are labeled `focused enough`; all others are `too blurry`.

#### usable vs not usable (NFP usability)
`usable`: sufficiently focused, contains cervical mucus, and appears in valid setup (finger or tissue).  
`not usable`: invalid setup (for example underwear/other), no visible cervical mucus, or clear off-domain/mistaken upload.

### Distribution and Overlap Notes
#### Multi-Label Overlap Example
At least `2` images were explicitly reported with both `blurry background` and `wrong setup`.

#### Contributor Concentration
For app uploads: `30` unique contributors were reported; contributor upload range includes a maximum of `215` images from one contributor.

#### Minimum Contributor Notes
Contributors min: `1`. Registered users min: `0`.

## Consent, Rights, and Retention
### Consent and Permission by Source
#### App Volunteers
Informed consent was collected with mixed scopes (some thesis-only, some broader research use). Exact counts by scope are pending; reuse decisions must be checked record-by-record.

#### Web
Images were publicly accessible at collection time, but licensing/copyright clarity is incomplete and requires source-level verification. No blanket redistribution right is assumed.

#### Professional-Shared Client Images
Used within the bachelor-thesis model-training context; future external reuse and publication permissions are not yet finalized.

#### FemTech Company Survey Images
Allowed uses documented as thesis and model training. Publication and third-party redistribution are treated as out of scope unless explicitly approved.

### Withdrawal and Retention
#### Withdrawal Process
App participants can request deletion via the email channel provided in consent documents. No withdrawal requests were used so far.

#### Retention Policy
Communicated primary retention deadline is March 31, 2026, unless optional extended consent is granted for longer use related to an open-source dataset objective.

### Legal and Compliance Context
#### GDPR Concept
A detailed GDPR concept is in place for Aura app collection, centered on transparency and consent.

#### Consent Completeness Status
Consent/legal basis is not uniformly complete across all sources, and a complete cross-source per-image permission matrix is not finalized. This blocks public release at this stage.

## Privacy, Security, and Risk
### Privacy Exposure
#### Potential Identifiers in Images
Some images include potentially identifying elements (for example faces, jewelry, and home bathroom backgrounds).

#### EXIF Metadata Status
EXIF metadata is currently preserved in stored files (for example potential timestamp/device/GPS fields). Analysis pipelines may use pixels only, but raw file metadata remains and is a privacy risk for any external sharing.

### Security Controls
#### Current Controls
- App access is password-protected.
- Server access requires university VPN plus server authentication.
- Local environments are password-protected.
- Transfer is protected via HTTPS.

#### Planned Controls
File-level encryption at rest is planned but not yet implemented.

### Harm and Misuse
#### Key Risks
- Privacy and re-identification harm in a sensitive, stigmatized topic area.
- Non-consensual or over-broad reuse of intimate imagery.
- Harm from misuse for fertility diagnosis or decision support without validated methods.

#### Current Mitigations
- Data minimization (no unnecessary personal identifiers such as real names in dataset records).
- Separation between account email identity and image records.
- Restricted access and non-public status.

## Example Images Policy
### Inclusion Rules
#### Whether Examples Are Included
Examples may be included in documentation (target: at least one example per category) only under strict consent and privacy controls.

#### Allowed Source for Examples
Only app-volunteer images are eligible for examples at this stage.

#### Required Consent for Examples
Examples are permitted only when explicit publication consent exists for the specific records.

#### De-identification for Examples
No mandatory automated de-identification pipeline is currently implemented. Before publication, manual privacy review is required and redaction/cropping should be applied when identifiable details are present. An optional ROI crop algorithm exists as a potential support step.

## Known Limitations
### Sampling and Distribution
#### Power-User Imbalance
A small subset of users contributes a large share of uploads.

#### Class Imbalance
Label/category distributions are strongly imbalanced and not close to 50/50 structures.

### Collection Bias
#### Instruction-Induced Bias
Users were explicitly asked to submit both high-quality and intentionally poor-quality images, so distribution may not reflect organic upload behavior.

### Documentation Gaps in v0.2
#### Currently Undocumented or Skipped Details
- Formal age-verification process details were not documented in this version.
- Detailed cleanup/removal rule counts were not documented in this version.
- Future release license planning is intentionally out of scope until consent coverage is resolved.

## Version History
### v0.1 to v0.2
#### v0.1 Baseline (August 2025)
Initial exploratory set with `161` web-sourced images, mostly from educational websites.

#### v0.2 Changes
- Added collected sources beyond web-only baseline.
- Added expanded annotation schema and label coverage.
- Increased use-case specificity toward usability filtering tasks.
- Some earlier web images became unavailable and could not be re-fetched.
