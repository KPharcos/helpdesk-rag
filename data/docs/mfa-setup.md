# Setting Up Multi-Factor Authentication (MFA)

Multi-factor authentication protects your account even if your password is stolen. All Brightfield accounts must have MFA. You register it during your first week; accounts without MFA are blocked from remote access after 7 days.

## Supported methods

- Authenticator app push notifications (recommended).
- Six-digit codes from the Authenticator app (works offline).
- SMS codes to a registered mobile number (backup only).
- Hardware security keys (FIDO2), issued to administrators and finance staff.

## Registering the Authenticator app

1. Install the Microsoft Authenticator app from your phone's app store. A personal phone is fine; the app does not give IT access to your personal data.
2. On your laptop, open https://mysignins.brightfield.example and sign in.
3. Select "Security info", then "Add sign-in method", then "Authenticator app".
4. Scan the QR code with the app and approve the test notification.
5. Add your mobile number as a backup method.

## Getting a new phone

Before you reset or trade in your old phone, sign in to https://mysignins.brightfield.example, add the Authenticator app on the new phone, and only then remove the old one. If the old phone is already gone, contact the IT Service Desk; after identity verification they can issue a temporary access pass valid for 8 hours so you can register the new phone.

## Unexpected MFA prompts

If you receive an MFA prompt you did not trigger, deny it and change your password immediately. Then report it to security@brightfield.example. Repeated unexpected prompts usually mean someone knows your password ("MFA fatigue" attack). Never approve a prompt just to make it stop.
