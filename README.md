I've been using [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for running various
machine learning loads, it is a very versatile platform with a lot of capabilities.

But for a lot of use cases with ComfyUI, I just feel that I need to write some python code.
Sometimes it's a little condtional logic I need here, sometimes it's a external API I want
to connect to, and for some cases simply the coolest model on huggingface is not ported to
ComfyUI yet. 

There are a decent size of community writing custom nodes for these purposes,
but... let's just say most of the code quality and engineering practices from the community
offers are a little bit less than desired, also the rise of AI coding agents combined with
a community that are gathered around a low-to-no-code platform aren't really going to produce
high quality code. For a lot of times I just ended up spending more time trying to dig a
bug in a pile of agent-wrote code when using these custom nodes.

And if you are like me, you probably feel more comfortable writing python code than wiring nodes
in a UI. So I decided to just go back to jupyter and directly call the pipelines that I need.

This is the repo that gathers all the helper functions that I use in my jupyter notebooks, with
some examples of the consuming notebooks. It is meant to be used by
[this container image](https://github.com/chaserhkj/containers/tree/main/service/jupyter)

No documentation is planned, use them at your own risk.
