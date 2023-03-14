# ARC Tutorial 
## Account creation
Create an ARC account by following instructions [here] (https://rcs.ucalgary.ca/How_to_get_an_account).

## Log into ARC
Once you have an ARC account, you can log in via terminal using SSH. If you use VS Code as your editor, there's the SSH extension that you can download, and follow their tutorial + the information below, you can now code in VS Code and avoid learning how to program via Terminal/SCP. However you'll still need basic linux terminal commands to run your code on ARC. If you choose to go the VS Code route, skip to the "VS Code Extension" section below.  

### iOS users
Search for "terminal" application. Open it and type: 
ssh <arc_acc_name>@arc.ucalgary.ca
Press Enter and you'll get a password prompt. Enter your password (synced with your UCalgary IT password). 

Example: 
<insert image> 

If you've successfully logged in, you should see this: 
<image> 

### Windows users
I'm not a windows user so there might be other ways but if you want to be able to follow the rest of this tutorial, I recommend downloading Git bash [download link] (https://git-scm.com/downloads). This comes with SSH terminal. Open the SSH terminal after you've installed git and follow the instructions above for iOS users. 

## Navigating your ARC account
In order for you to do things using the SSH terminal, you'll need to learn some simple linux terminal commands. There are many resources on the internet if you google "linux commands". Here is the minimal list that you'll need: 


### Exercise 
Let's set up a simple file structure on ARC for running code. Then we'll see how to transfer files. 
.. some stuff 

Now that we have the folder structure set up, how can we transfer our data/code onto ARC to run? There's many ways one can do file transfer. Three common options: 
1. Use a software like FileZilla. This is a graphical tool so it might be easier to pick up for non-programmer background individuals. 
2. Remote directory mounting. Google it. 
3. Use the Secure Copy Protocol (scp). This is a shell protocol that is well suited for the terminal. I like to use this because I'm comfortable with working from the terminal and I don't have to download/install anything extra. If you're on iOS, you should have this built in. If you downloaded Git, you should have this. 

### File Transfer via SCP
blah blah. easy. 

## VS Code Extension 


## Other ARC references/resources 
* Maria's Github ARC [Guide] ().
 

