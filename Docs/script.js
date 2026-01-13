async function enviarTexto() {
    const texto = document.getElementById("texto").value;
    const resultado = document.getElementById("resultado");

    if (!texto.trim()) {
        resultado.innerHTML = "⚠️ Debes escribir un texto";
        return;
    }

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ texto: texto })
        });

        if (!response.ok) {
            throw new Error("Error en la petición");
        }

        const data = await response.json();

        // Construir HTML del resultado
        let html = `<strong>Resultado:</strong> ${data.sentimiento}<br><br>`;
        html += `<strong>Probabilidades:</strong><br>`;

       if (data.texto_traducido) {
        html += `<strong>Texto traducido:</strong><br>`;
        html += `<em>${data.texto_traducido}</em><br><br>`;
     }
     html += `<strong>Probabilidades:</strong><br>`;

        for (const [sentimiento, prob] of Object.entries(data.probabilidades)) {
            html += `${sentimiento}: ${(prob * 100).toFixed(2)}%<br>`;
        }

        resultado.innerHTML = html;

    } catch (error) {
        resultado.innerHTML = "❌ Error al analizar el texto";
        console.error(error);
    }
}
