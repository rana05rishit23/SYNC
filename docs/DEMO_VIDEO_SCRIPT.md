# 3-Minute Demo Video Script: Project Udyam Digital

> **Submission Track:** YTL Hackathon 2026 — Track A (Project Udyam Digital)  
> **Team:** ONE8 (Vipul Kumar & Rishit Rana)  
> **Platform:** Accel Business Self-Service Onboarding Platform  
> **Target Video Length:** Exactly 3:00 (180 Seconds)  
> **Format:** Split screen / Picture-in-Picture (Presenter + Mobile Screen Recording)

---

## Video Production & Screen Recording Guide

| Time | Slide / Screen | Speaker | Visual & Actions on Screen |
|---|---|---|---|
| **0:00 – 0:25** | **Slide 1 & 2** (Problem) | **Speaker 1** (Vipul) | Hook the judges. Show 17.6 day lag and 93% funnel drop-off. |
| **0:25 – 1:05** | **Live App** (Step 1: Advisor) | **Speaker 2** (Rishit) | Type natural chat in Hindi/Hinglish. Show instant segment detection. |
| **1:05 – 1:35** | **Live App** (Step 2: Quote) | **Speaker 2** (Rishit) | Toggle 12m vs 24m term. Add/remove security add-on. Total updates live. |
| **1:35 – 2:10** | **Live App** (Step 3: Smart KYC) | **Speaker 1** (Vipul) | Show Consent Gate (DPDP Act). Upload certificate. Instant OCR & registry check. |
| **2:10 – 2:35** | **Live App & Slide 5** (Step 4 & Arch) | **Speaker 2** (Rishit) | Show Order Tracker timeline, then the deterministic core architecture. |
| **2:35 – 3:00** | **Slide 8 & 10** (Impact & Close) | **Speaker 1** (Vipul) | ROI metrics, 215 automated tests passing, pilot readiness. Strong closing. |

---

## Detailed Chronological Script

### Section 1: The Problem & Vision (0:00 – 0:25)
*Visual: Slide 2 on screen, camera on Speaker 1 in top-right corner.*

> **Speaker 1 (Vipul):**  
> *"Namaste and hello judges. In India today, over 63 million MSMEs want to digitize, but telco onboarding is broken. It takes an average of **17.6 days**, costs ₹15,000 per acquisition, and **over 93% of leads drop off** before a quote is even accepted.  
> 
> Why? Because small business owners don't understand complex telecom product codes, have no time for in-person sales visits, and get stuck in endless document chases.  
> 
> We built **Project Udyam Digital** by Accel Business: an intelligent, mobile-first platform that takes a small business from discovery to verified onboarding **in one sitting, on their phone, in their own language.**"*

---

### Section 2: Conversational Advisor & Segment Detection (0:25 – 1:05)
*Visual: Cut to Mobile UI screen recording showing Step 1 (Advisor Chat).*

> **Speaker 2 (Rishit):**  
> *"Let's see it live. Imagine Ramesh, who runs a chartered accountancy firm with 8 staff in Pune. He visits our web app on his mobile.  
> 
> Instead of browsing through 25 confusing SKUs, Ramesh simply chats in Hindi or English:  
> `\"Meri CA firm hai Pune me, 8 log kaam karte hain, accha internet aur security chahiye.\"`  
> 
> Watch what happens: our AI edge, powered by Groq, instantly detects the language, extracts the business profile, and passes it to our **deterministic rules engine**.  
> 
> Notice: the AI never guesses products or invents prices. Our rules engine selects the optimal **Professional Office Bundle** with Biz Fibre 300 and Secure Internet Gateway. The advisor explains the recommendation back in plain Hindi with complete numeric accuracy."*

---

### Section 3: Transparent Quote & Flexible Terms (1:05 – 1:35)
*Visual: Click "View Customized Quote" -> Step 2 (Quote Screen).*

> **Speaker 2 (Rishit):**  
> *"With one tap, Ramesh enters the Quote step.  
> 
> Here, there are no surprise fees or hidden riders. He sees one transparent monthly bill: **₹2,099/month**.  
> 
> Watch the term toggle: switching between 12 months and 24 months recalculates the discount dynamically based on our catalogue rules—40% margin discount for 12 months, and 80% for 24 months.  
> 
> If Ramesh toggles an add-on like Static IP or Cloud Backup, our catalogue dependency engine validates eligibility in real-time. If he is satisfied, he clicks **'Proceed to Digital KYC'**."*

---

### Section 4: DPDP Smart KYC & Registry Verification (1:35 – 2:10)
*Visual: Step 3 (Consent & KYC Upload).*

> **Speaker 1 (Vipul):**  
> *"Now comes the biggest point of friction in telecom: KYC. We designed our KYC pipeline strictly compliant with the **DPDP Act 2023**.  
> 
> First: **Privacy by Construction**. Ramesh sees an affirmative consent gate detailing exactly why his data is collected. The backend blocks all upload endpoints with HTTP 403 until consent is affirmatively given.  
> 
> Now, Ramesh uploads his GST certificate. Behind the scenes:  
> 1. Our OCR engine cleans diagonal watermarks and runs deskew preprocessing.  
> 2. Even if OCR confuses a `T` for a `7`, our **Mod-36 structured repair algorithm** restores the valid GSTIN.  
> 3. It checks the registry in real-time: verifying business status, legal name, and rejecting cancelled or suspended entities on the spot.  
> 4. All identifiers are masked (`37IHK••••••••1ZS`), protecting customer PII in logs and UI."*

---

### Section 5: Order Tracking & Architecture (2:10 – 2:35)
*Visual: Step 4 (Order Tracker), then transition to Architecture Diagram (Slide 5).*

> **Speaker 2 (Rishit):**  
> *"Once KYC succeeds, Ramesh receives an instant Order Tracker timeline with transparent milestones, replacing 3 to 4 anxious follow-up calls.  
> 
> Under the hood, our architecture enforces a strict boundary: **AI at the Edge, Deterministic Rules at the Core**.  
> - Language detection and conversational interaction happen through ultra-low-latency Groq LLMs.  
> - But every single pricing computation, bundle validation, and catalogue dependency is executed by 100% deterministic, auditable Python code. It is mathematically impossible for the LLM to hallucinate an illegal discount or unauthorized product."*

---

### Section 6: Business Impact & Pilot Readiness (2:35 – 3:00)
*Visual: Slide 8 and Slide 10, showing the metric dashboard and test results.*

> **Speaker 1 (Vipul):**  
> *"The business impact is transformative:  
> - Cuts customer acquisition cost from ₹15,000 to under ₹5,000.  
> - Compresses onboarding time from 17.6 days down to **under 15 minutes**.  
> - Eliminates the 15% rule violations found in historical manual orders.  
> 
> Our codebase is production-hardened: **215 automated tests passing** across 47 official KYC specimens, zero external security vulnerabilities, and complete DPDP compliance.  
> 
> Project Udyam Digital empowers India's MSMEs to describe, quote, verify, and track—all in one sitting.  
> Thank you!"*

---

## Quick Recording Checklist for the Team

- [ ] **Video Resolution:** 1080p (1920x1080) at 30/60 fps.
- [ ] **Audio Quality:** Use a clean lapel or USB microphone; ensure zero echo or background hum.
- [ ] **Screen Recording:** Capture `frontend/index.html` in Chrome Mobile Device Mode (e.g., iPhone 14 Pro or Pixel 7 layout) for an authentic mobile-first look.
- [ ] **Timing Check:** Keep stopwatch running during rehearsal to verify speaker pace stays within 180 seconds.
