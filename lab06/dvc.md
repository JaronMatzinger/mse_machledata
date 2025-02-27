# Data Version Control with DVC

## Installation

We recommend installing `dvc` using `conda` as this lab comes with
its own `conda` environment.

```raw
conda create -f env.yaml
conda activate mlops-lab-06
conda install dvc
```

Don't forget to activate your new environment

```raw
conda activate mlops-lab-06
```

If you prefer installing it in a different manner, please refer to [the dvc documentation](https://dvc.org/doc/install/).

## Data Management with DVC

DVC lets you capture the versions of your data (and models) in Git commits, while storing the actual data outside of the repository.
It also provides you with a mechanism to switch between the different data contents. When used regularly and diligently, the result is a single history of data, code, and ML models.

DVC uses what they refer to as _versioning through codification_. You produce _metafiles_ once, which describe what datasets, ML artifacts, etc. to track. This metadata can be put in Git in place of large files.
Once that's done, you can use DVC to create snapshots of the data, restore previous version, reproduce experiments - and much more.

There are _many_ features and uses cases for DVC. We will highlight two of them, and let you explore the remaining ones, [which you can explore in the DVC docs](https://dvc.org/doc/use-cases).

We will start with the basics of DVC and afterwards show you to manage data with DVC. To conclude our DVC adventure, we will show you how to create a dataset registry, a centralized repository for all your data.

### DVC 101

DVC is modelled after, and built on top of, git, and a _DVC project_ always goes together with a git repository.

To create a new DVC project, you first have to create a git repository (i.e. use `git init`).
After initializing the git repository, you can initialize the DVC project. You do this using [`dvc init`](https://dvc.org/doc/command-reference/init).

```shell
dvc init
```

Upon running `dvc init`, a [directory `.dvc`](https://dvc.org/doc/user-guide/project-structure/internal-files) is created (similar to how `git init` creates `.git`). `.dvc` creates configuration files and a cache for project data. The details are not important for our purposes.

Should you be unhappy with DVC, you can use `dvc destory` to remove all DVC-specific files from the directory. This is synonymous with "deleting" the DVC project.

Suppose you have a big file, e.g. `hour.csv`. Because it is so big, we cannot check it into Git directly. To track this file with DVC instead, we can use `dvc add`:

```shell
dvc add hour.csv
```

This will do the following:

1. The original file is moved into the cache in `.dvc`.
2. A pointer, named `<file name>.dvc` is created in lieu of the original file.
3. The path is added to `.gitignore` to prevent Git from tracking the file.

In the end, the situation looks as depicted in the image below:

![alt text](imgs/dvc_versioning.png)
(Image taken from [here](https://dagshub.com/blog/getting-started-with-dvc/).)

What happens if you modify a file tracked by DVC? When you change `hour.csv` and later add this change to DVC, DVC
will copy the new version to the cache and update the pointer.

#### Remotes

The description above might have you left wondering about how you can share the different file versions with your collaborators.
After all, the cache is local. This is where remotes come in. In their own words,

> DVC remotes provide access to external storage locations to track and share your data and ML models. Usually, those will be shared between devices or team members who are working on a project. For example, you can download data artifacts created by colleagues without spending time and resources to regenerate them locally.

There are two main uses of remote storage:

1. synchronization of large files and directories tracked by DVC
2. centralization of data storage for sharing and collaboration

DVC supports a range of different providers and storage types such as `LFS`, `S3`, `G-Drive`, [and more](https://dvc.org/doc/user-guide/data-management/remote-storage).

#### Adding and tracking data

There are many ways of adding data to your DVC project. Here are a few:

First off, there's `dvc add`, which you've already learnt about above. This tracks data files or directories with DVC. If you think back to the previous part
on `git lfs`, this step corresponds to `git lfs track`.

Then, there are two commands to add external data to your DVC project. There commands come in two flavours, `get` and `import`.
`get` commands do **not** track the downloaded data files, while `import` commands **do** track the files.

Then, there are also `-url` and non `-url` commands. The commands with a `-url` suffix can be used to add files that are external to DVC or git (i.e., they are not already tracked by DVC or Git), while the commands without a suffix can only be used with

- `dvc get`: Download a file or directory tracked by DVC or by Git into the _current working directory_. This command does **not** track the downloaded files.
- `dvc get-url`: Download a file or directory from a supported URL, for example `s3://`, `ssh://`, and other protocols, into the local file system. This is comparable to a command like `wget` or `curl`.
- `dvc import`: Download a file or directory tracked by another DVC or Git repository into the workspace, and track it (an import `.dvc` file is created).
- `dvc import-url`: Download a file or directory from a supported URL, for example `s3://`, `ssh://`, and other protocols, and track it.

A small comment on tracking. When you track data (`dvc add`, `dvc import`), DVC will compute hashes which it later uses to check whether modifications have been made. If you have a lot of data in your repository, this will eventually become very expensive. So, instead of tracking all the files individually, you might be tempted to instead track e.g. a ZIP archive or a tarball of the data. Note, however, that you will lose the ability to efficiently track changes that were made to your data if you end up using tarballs, because the tarball hash will be different every time - you might as well use LFS in this case. (There are still cases where you maybe want to `add` tarballs to DVC - more on this later.) Don't forget that you should only commit changes to the data that you want to keep, e.g. large augmentations or newly added samples, so you won't run `dvc add` very frequently.

#### A note on "workspaces"

If you read the DVC documentation, you will come across the concept of a _DVC workspace_. This is again something that
DVC borrows from Git. In Git,

> A repository can have zero (i.e. bare repository) or one or more worktrees attached to it. One "worktree" consists of a "working tree" [= the tree of actual checked out files] and repository metadata, most of which are shared among other worktrees of a single repository, and some of which are maintained separately per worktree (e.g. the index, HEAD and pseudorefs like MERGE_HEAD, per-worktree refs and per-worktree configuration file).

A DVC workspace is analogous to a Git working tree. It is simply the currently visible version of the project.

## DVC hands-on

Enough theory, let's get started. First, create a git repository on github.zhaw.ch.
Then, in your git repository, run

```shell
dvc init
```

To see the files that DVC created, run `git status`. The output should look something like the following:

```raw
On branch main
Your branch is up to date with 'origin/main'.

Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        new file:   .dvc/.gitignore
        new file:   .dvc/config
        new file:   .dvcignore
```

Commit these files to git.

```shell
git commit -m "Initialize DVC`
```

Now we are ready to add data to our DVC repository. Create the `data` directory and change into it.

```shell
mkdir data
cd data
```

We'll again use the `102flowers` dataset. Download the tarball using DVC but don't track it.
The download URL is

```raw
https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
```

<details>
    <summary>Solution</summary>

```shell
dvc get-url <URL>
```

</details>

Once that's completed, extract the tarball

```shell
# Extract te contents of the tarball.
tar -xvf 102flowers.tgz

# Rename the output folder
mv jpg 102flowers

# Delete the tarball
rm 102flowers.tgz
```

You should now have a folder `data/102flowers`.

Now, track this folder with DVC!

<details>
    <summary>Solution</summary>

```shell
dvc add data/102flowers
```

</details>

Let's inspect the resulting "pointer":

```shell
cat 102flowers.dvc
```

```raw
outs:
- md5: 56eae55ec88caaf5b57e6ae9314484d4.dir
  size: 346809251
  nfiles: 8189
  hash: md5
  path: 102flowers
```

As you can see, there's the directory hash (`md5`),
the `size` and number of files `nfiles` along with the directory path.

Let's commit the pointer and `.gitignore` to git and push it to the remove.

<details>
    <summary>Solution</summary>

```shell
git add data/.gitignore data/102flowers.dvc
git commit -m "Add 102flowers"
git push
```

</details>

Last, but definitely not least, let's push the data to a remote location. As we've mentioned above, DVC supports many
storage locations. For the sake of simplicity, we will use a local path.

Create a temporary directory somewhere on your system. We opt for the temporary filesystem `/tmp`:

```shell
mkdir /tmp/dvcdata
```

Adding a remote in DVC follows the same syntax as Git:

```shell
dvc remote add <name> <url>
```

where url can also be a local path.
Add the temporary directory you create before as a remote.
Make sure to use the _absolute_ path (`/very/long/path`) and not a relative one (`./my/path` or `../my/path`).
Relative paths work, but may lead to surprises.

_Hint_: You can designate a remote the _default remote_ by adding `-d` or `--default` to the command.

<details>
    <summary>Solution</summary>

```shell
dvc remote add temporary /tmp/dvcdata
```

</details>

This will add an entry to `.dvc/config`, which you have to commit to your git repository to
persist.

```shell
git add .dvc/config
git commit -m "Add remote to config"
git push
```

To push your data to the remote, simply run

```shell
dvc push
```

Now, take a leap of faith and delete your local copy of the repository and
clone it again from github. Note how there is no data in `data/`!
To "download" the data from the remote, run

```shell
dvc pull
```

---

That's it, you've mastered the basics of DVC! Don't delete this repository just yet, we will
use it again.
