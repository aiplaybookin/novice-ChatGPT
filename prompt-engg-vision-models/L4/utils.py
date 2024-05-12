def generate_image(
    image,
    mask,
    prompt,
    pipe,
    seed,
    guidance_scale=7.5,
    negative_prompt="low resolution, ugly",
    num_inference_steps=120,
    strength=1,
):
    # resize for inpainting
    w, h = image.size
    in_image = image.resize((512, 512))
    in_mask = mask.resize((512, 512))

    generator = torch.Generator(device).manual_seed(seed)

    result = pipe(
        image=in_image,
        mask_image=in_mask,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        strength=strength,
        generator=generator,
    )
    result = result.images[0]

    return result.resize((w, h))


def show_generated_image(generated_image):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(generated_image, interpolation="nearest")
    plt.tight_layout()


def make_birthday_card(img, filename=None):
    h, w, _ = np.array(generated_image).shape
    x = [randint(50, w) for x in range(50)]
    y = [randint(50, h) for x in range(50)]
    color_list = [
        "darkmagenta",
        "orangered",
        "greenyellow",
        "lightpink",
        "orange",
        "gold",
        "forestgreen",
        "blue",
        "mediumturquoise",
        "magenta",
        "dodgerblue",
    ]
    c = [choice(color_list) for x in range(50)]

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.text(
        x=(w / 12),
        y=((9 * h) / 10),
        s="Happy Birthday!",
        c="tomato",
        path_effects=[pe.withStroke(linewidth=3, foreground="oldlace")],
        fontsize=58,
        fontname="serif",
    )
    plt.scatter(x=x, y=y, s=50, c=c, marker=(5, 1), alpha=0.7)
    plt.axis("off")
    if filename:
        plt.savefig(filename)
    plt.show()
