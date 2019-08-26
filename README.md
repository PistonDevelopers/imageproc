imageproc
====

[![crates.io](https://img.shields.io/crates/v/imageproc.svg)](https://crates.io/crates/imageproc)
[![Build Status](https://travis-ci.org/image-rs/imageproc.svg?branch=master)](https://travis-ci.org/image-rs/imageproc)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/image-rs/imageproc/blob/master/LICENSE)

An image processing library, based on the [image](https://github.com/image-rs/image) library. There may initially
be overlap between the functions in this library and those in image::imageops.

This is very much a work in progress. If you have ideas for things that could be done better, or new features you'd like to see, then please create issues for them. Nothing's set in stone.

[API documentation](http://docs.rs/imageproc)

# Goals

A performant, well-tested, well-documented library with a consistent API, suitable for use as the basis of computer vision applications or graphics editors.

# Non-goals

Maximum genericity over image storages or formats, or support for higher-dimensional images.

Full blown computer vision applications (e.g. face recognition or image registration) probably also belong elsewhere, but the line's a bit blurred here (e.g. is image in-painting an image processing task or a computer vision task?). However, worrying about how to structure the code can probably wait until we have more code to structure...

# How to contribute

All pull requests are welcome. Some specific areas that would be great to get some help with are:

* New features! If you're planning on adding some new functions or modules, please create an issue with a name along the lines of "Add [feature name]" and assign it to yourself (or comment on the issue that you're planning on doing it). This way we'll not have multiple people working on the same functionality.
* Performance - profiling current code, documenting or fixing performance problems, adding benchmarks, comparisons to other libraries.
* Testing - more unit tests and regression tests. Some more property-based testing would be particularly nice.
* APIs - are the current APIs hard to use or inconsistent? Some open questions: Should we return Result types more often? How should functions indicate acceptable input image dimensions? Should we use enum arguments or have lots of similarly named functions? What's the best way to get concise code while still allowing control over allocations?
* Documentation - particularly more example code showing what's currently possible. Pretty pictures in this README.
* Feature requests - are there any functions you'd like to see added? Is the library currently unsuitable for your use case for some reason?

# Unsafety

This library contains quite a few uses of `unsafe`. Each `unsafe` block should have a comment explaining both its benefit and correctness. Most are to avoid bounds checks when indexing into arrays. PRs to remove uses of `unsafe` without regressing performance will be enthusiastically accepted. PRs which remove `unsafe` at the cost of regressing performance will be rejected unless they come with an explanation of why the `unsafe` use in question is problematic.