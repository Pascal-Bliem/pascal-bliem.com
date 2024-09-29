import React, { useEffect } from "react";
import styles from "./VecBrain.module.scss";
import Navbar from "../../layout/Navbar";
import Footer from "../../layout/Footer";

const Diarysta = () => {
  useEffect(() => {
    document.title = "Vec Brain LLM - Pascal Bliem";
  }, []);

  return (
    <div className={styles.page}>
      <Navbar fullyHideHeight={0} initialFullyHide={false} />
      <div className={`container`}>
        <div className="row">
          <div className="col s12 m10 offset-m1 l8 offset-l2 center-align">
            <h1 className={styles.vecBrainTitle}>Vec Brain LLM</h1>
          </div>
        </div>
        <div className="row">
          <div
            className={`col l8 offset-l2 m10 offset-m1 s12 ${styles.postContentContainer}`}
          >
            <p>
              Vec Brain is a personal knowledge base app based on a large
              language model (LLM) and retrieval augmented generation (RAG).
              Users can add information from different sources to their personal
              knowledge store. The assistant will answer based on the
              information added and cite sources to let you know from which of
              your documents this information came from. If it can't retrieve
              information that is similar enough to the users query, it will
              give a general answer based on its own knowledge. The information
              can be kept either persistently in the{" "}
              <a href="https://www.pinecone.io/">Pinecone</a> vector database or
              ephemerally in memory.
            </p>
            <p>
              Check out the{" "}
              <a href="https://github.com/Pascal-Bliem/vec-brain-llm">
                Vec Brain LLM <span style={{fontWeight: "bold"}}>GitHub Repository</span> 
              </a>
              !
            </p>
            <div className={styles.vecBrainImg1}>
              <img
                src="https://raw.githubusercontent.com/Pascal-Bliem/vec-brain-llm/refs/heads/main/demo.gif"
                alt="Asking Vec Brain about knowledge you stored in it"
              />
            </div>
            <h3>How it works</h3>
            <p>
              The conversational ability of the agent is based on a large
              language model (LLM). The user can ask the agent to store
              information (e.g. from websites or documents) in the knowledge
              base. The agent extracts the information from the provided
              sources, chunks it into documents, creates vector embeddings for
              these documents. These vector embeddings are a way to
              mathematically encode the meaning of the content of the documents
              in a way that is suitable for similarity search. The agent then
              saves these embeddings in a vector store. When the user has a
              question, embeddings are created for that question as well and a
              similarity search over the vector store is performed. In this
              search, the vector representing the user query is compared to
              vectors in the database and documents with vectors that point in a
              similar direction in the high-dimensional vector space are
              retrieved. If similar documents are found, the agent will base its
              answer on the content of these documents and cite the sources. If
              no similar documents are found, it will give a general answer
              based on the underlying LLMs training data.
            </p>
            <div className={styles.vecBrainImg2}>
              <img
                src="https://raw.githubusercontent.com/Pascal-Bliem/vec-brain-llm/refs/heads/main/flowchart.png"
                alt="The application flow"
              />
            </div>
          </div>
        </div>
      </div>
      <Footer acknowledgements={<div />} />
    </div>
  );
};

export default Diarysta;
