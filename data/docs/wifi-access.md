# Office Wi-Fi Networks

Brightfield offices provide three Wi-Fi networks. Use the right one for your device to get access to internal systems and to keep the corporate network secure.

## Available networks

- **Brightfield-Corp**: for company laptops only. Connects automatically using the device certificate; no password needed. Gives full internal access.
- **Brightfield-BYOD**: for employees' personal phones and tablets. Sign in with your work email and password. Internet access only, plus email and chat.
- **Brightfield-Guest**: for visitors. Internet access only. A daily password is shown on the reception screens and is valid for 24 hours.

## Connecting a company laptop

Company laptops connect to Brightfield-Corp automatically. If the laptop does not connect, check that the date and time are correct, then restart it. If it still fails, the device certificate may have expired; see the VPN article section on certificates or contact the IT Service Desk.

## Connecting a personal phone

1. Select Brightfield-BYOD in your phone's Wi-Fi settings.
2. When asked for an EAP method choose PEAP, and for phase 2 choose MSCHAPV2 (Android only).
3. Enter your work email as identity and your Brightfield password.
4. Accept the certificate for "radius.brightfield.example".

After you change your password, update it on your phone too; an old saved Wi-Fi password is the most common cause of account lockouts.

## Guests and visitors

Visitors should use Brightfield-Guest. Never share your own credentials with a guest or connect a visitor's laptop to Brightfield-Corp or a network port. For events with more than 20 guests, ask the IT Service Desk for a dedicated event network at least 3 working days in advance.

## Slow or unstable Wi-Fi

Move closer to an access point, or use the wired network ports at desks (company laptops only). Report persistent dead spots to the IT Service Desk with the floor and room number.
