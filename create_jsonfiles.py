import os
import json
import logging
import pyodbc

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def generate_json_labels_from_sql(
    pdf_dir="dataset/train/pdfs",
    label_dir="dataset/train/labels",
    server="localhost\SQLEXPRESS",
    database="DiplomeDB",
    user="JIHANEPC\hp",
    password=""
):
    os.makedirs(label_dir, exist_ok=True)

    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={server};"
        f"DATABASE={database};"
        "Trusted_Connection=yes;"
    )

    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
    except Exception as e:
        logging.error(f"Connexion à SQL Server échouée: {e}")
        return

    for file in os.listdir(pdf_dir):
        if file.endswith("_DIPLOME.pdf"):
            try:
                num_inscription = file.split("_")[0]

                cursor.execute("""
                    SELECT d.TypeDiplome, d.Etablissement, d.Specialite, d.AnneeDiplome,
                           c.Nom, c.Prenom
                    FROM diplome d
                    JOIN cin c ON d.NumInscription = c.NumInscription
                    WHERE d.NumInscription = ?
                """, num_inscription)
                row = cursor.fetchone()

                if not row:
                    logging.warning(f"Aucune donnée trouvée pour {num_inscription}")
                    continue

                data = {
                    "gt_parse": {
                        "NumInscription": num_inscription,
                        "Nom": row.Nom,
                        "Prenom": row.Prenom,
                        "TypeDiplome": row.TypeDiplome,
                        "Etablissement": row.Etablissement,
                        "Specialite": row.Specialite,
                        "AnneeDiplome": row.AnneeDiplome
                    }
                }

                json_path = os.path.join(label_dir, f"{num_inscription}_DIPLOME.json")
                with open(json_path, "w", encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)

                logging.info(f"✅ JSON créé pour {num_inscription}")

            except Exception as e:
                logging.error(f"Erreur pour {file}: {e}")

if __name__ == "__main__":
    generate_json_labels_from_sql()
