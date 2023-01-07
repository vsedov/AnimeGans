#
# def get_data_from_e621(self) -> None:
#     """Download data from e621"""
#     os.chdir(self.path)
#     for tag in self.tags:
#         os.system(
#             f'gallery-dl --range 1-100 "https://e621.net/posts?tags={tag}"'
#         )
#
# def get_data_from_gelbooru(self) -> None:
#     """Download data from gelbooru"""
#     os.chdir(self.path)
#     for tag in self.tags:
#         os.system(
#             f'gallery-dl --range 1-100 "https://gelbooru.com/index.php?page=post&s=list&tags={tag}"'
#         )
#
class Anime:
    def __init__(
        self, name: str, tags: list, path: str, create_dataset: bool = False
    ):
        self.name = name
        self.tags = tags
        self.path = path
        self.create_dataset = create_dataset
        self.danbooru_path = f"{self.path}/gallery-dl/danbooru"

    def get_data_from_danbooru(self) -> None:
        """Download data from danbooru"""
        os.chdir(self.path)
        if os.path.isdir(f"{self.path}/gallery-dl"):
            print("Path exists")

        for tag in self.tags:
            print(tag)
            if not os.path.isdir(f"{self.danbooru_path}/{tag}"):
                os.system(
                    'gallery-dl --range 1-100 --filter "extension in ("jpg", "png") '
                    + f"https://danbooru.donmai.us/posts?tags={tag}"
                )

    # def dataset(self) -> None:
    #     return AnimeDataset(self.path, self.tags)


if __name__ == "__main__":
    x = Anime("Current", get_tags_from_danbooru(), current_path, True)
    x.get_data_from_danbooru()
    # dataset = x.dataset()
    # print(dataset)
