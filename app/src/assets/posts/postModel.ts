export type TagsType =
  | "Data Science & AI/ML"
  | "Web Development"
  | "Learning"
  | "Non-Tech";

export default class Post {
  title: string;
  subtitle: string;
  publishDate: Date;
  titleImageUrl: string;
  titleImageDescription: string;
  tags: TagsType[];
  content: string;

  constructor(
    title: string,
    subtitle: string,
    publishDate: Date,
    titleImageUrl: string,
    titleImageDescription: string,
    tags: TagsType[],
    content: string
  ) {
    this.title = title;
    this.subtitle = subtitle;
    this.publishDate = publishDate;
    this.titleImageUrl = titleImageUrl;
    this.titleImageDescription = titleImageDescription;
    this.tags = tags;
    this.content = content;
  }
}
