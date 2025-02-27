# GCP Group Setup Guide

Google Cloud Platform (GCP) provides a framework for building and managing applications and services. A `GCP Project` acts as the primary entity under which we manage the group's users, billing and services. This guide shows you how to set up your group and redeem your credits. **_Note:_ Only one person per group needs to complete these steps!**

## Create Project

The first step is to create a project where you can manage users and resources.

1. Head over to [https://console.cloud.google.com](https://console.cloud.google.com) and log in. You will need to log in with a google account _not_ your zhaw account.
1. Click on `create or select a project`.
![login](imgs/login.png)
1. Create a project by selecting `new project` on the top right and give it a fitting name and click `create`. **_Note:_** this might take a moment.
![new_project](imgs/new_project.png)
1. When your project is ready, select it to go to the `GCP dashboard`.

## Add users to Project

Once the project is created, you can invite your group members.

1. In the dashboard, open the `navigation Menue` (the three lines on the top left).
![dashboard](imgs/dashboard.png)
1. Select `IAM and admin` to add users
![iam admin](imgs/nav_menue_add.png)
1. Fill in your groupmembers Email and assign a role before adding them. For simplicitys sake, we recommend you add them as owner.
![add user](imgs/add_user.png)

## Redeem Credits

To use GCP resources you will need to redeem your group's vouchers and link the credits to your project.

1. Open [https://console.cloud.google.com/edu](https://console.cloud.google.com/edu) to get to the credit application.
1. Verify that you are logged in with the right accound and enter your voucher. Click `Accept and Continue` to redeem it. **_Note:_** you will need to repeat this process for every voucher.
![credit application](imgs/credit_app.png)
1. Go back to `GCP` and select your project.
1. To link the credits to the project, open the `Navigation Menue` and select `Billing`
![nav menue billing](imgs/nav_menue_bill.png)
1. Select the `Billing Account for Education` and set the account.
![billing account](imgs/set_billing_account.png)

## Verify Credits

The last step is to verify that the credits are available inside your project.

1. Go back to `GCP` and select your project.
1. To link the credits to the project, open the `Navigation Menue` and select `Billing`
![nav menue billing](imgs/nav_menue_bill.png)
1. On the billing page, scroll down and select `Credits`
![credits](imgs/billing_nav_credits.png)
1. You should see a credit names `Machine Learning Operations`
![alt text](imgs/credits_show.png)
