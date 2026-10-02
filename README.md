
# Academic Pages

![pages-build-deployment](https://github.com/academicpages/academicpages.github.io/actions/workflows/pages/pages-build-deployment/badge.svg)

Academic Pages is a Github Pages template for academic websites.


# Getting Started

1. Register a GitHub account if you don't have one and confirm your e-mail (required!)
1. Click the "Use this template" button in the top right.
1. On the "New repository" page, enter your repository name as "[your GitHub username].github.io", which will also be your website's URL.
1. Set site-wide configuration and add your content.
1. Upload any files (like PDFs, .zip files, etc.) to the `files/` directory. They will appear at https://[your GitHub username].github.io/files/example.pdf.  
1. Check status by going to the repository settings, in the "GitHub pages" section
1. (Optional) Use the Jupyter notebooks or python scripts in the `markdown_generator` folder to generate markdown files for publications and talks from a TSV file.

See more info at https://academicpages.github.io/

## Running Locally

Use Ruby 3.3 and Bundler to preview the site before pushing changes to GitHub.
From the root directory of this repository, run:

```sh
bundle config set --local path vendor/bundle
bundle install
bundle exec jekyll serve --livereload
```

Open <http://127.0.0.2:4001/>. Changes rebuild the site and reload the browser;
restart the server after changing `_config.yml`. Press `Ctrl+C` to stop it.
The preview uses the loopback address `127.0.0.2` and port `4001` from `_config.yml`.
LiveReload uses `127.0.0.2:35729`, so this site can run alongside
`robotic-manipulation-notebook` on `127.0.0.1`. Jekyll 3.10 requires a command-line
`--livereload-port` option to change that port; it ignores the YAML setting.

On the configured local machine, `bundle` is available through `~/.local/bin/bundle`,
which uses the Ruby environment shared with `robotic-manipulation-notebook`.
Each project's gems are installed separately in its own `vendor/bundle` directory.
To use Ruby or RubyGems directly, run `conda activate robotic-manipulation-notebook`.

On another Linux machine with Conda, create the Ruby environment first:

```sh
conda create -n jekyll --override-channels -c conda-forge \
  ruby=3.3.6 gcc_linux-64=13 gxx_linux-64=13 make pkg-config
conda activate jekyll
```

The generated site (`_site`), local gems (`vendor/bundle`), Bundler settings
(`.bundle`), and resolved dependency versions (`Gemfile.lock`) are ignored by Git.
Keep the local lockfile to reuse the same dependency versions on subsequent installs.

# Maintenance 

Bug reports and feature requests to the template  should be [submitted via GitHub](https://github.com/academicpages/academicpages.github.io/issues/new/choose). For questions concerning how to style the template, please feel free to start a [new discussion on GitHub](https://github.com/academicpages/academicpages.github.io/discussions).

This repository was forked (then detached) by [Stuart Geiger](https://github.com/staeiou) from the [Minimal Mistakes Jekyll Theme](https://mmistakes.github.io/minimal-mistakes/), which is © 2016 Michael Rose and released under the MIT License (see LICENSE.md). It is currently being maintained by [Robert Zupko](https://github.com/rjzupkoii) and additional maintainers would be welcomed.

## Bugfixes and enhancements

If you have bugfixes and enhancements that you would like to submit as a pull request, you will need to [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) this repository as opposed to using it as a template. This will also allow you to [synchronize your copy](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork) of template to your fork as well.

Unfortunately, one logistical issue with a template theme like Academic Pages that makes it a little tricky to get bug fixes and updates to the core theme. If you use this template and customize it, you will probably get merge conflicts if you attempt to synchronize. If you want to save your various .yml configuration files and markdown files, you can delete the repository and fork it again. Or you can manually patch.
