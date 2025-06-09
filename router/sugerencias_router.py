from fastapi import APIRouter
from pydantic import BaseModel
from utils.puntaje import score_phrase
from controller.sugerencias_controller import get_suggestions

router = APIRouter(prefix="/suggest", tags=["suggest"])

class SuggestReq(BaseModel):
    phrase: str
    pattern: list[str]

@router.post("", response_model=list[str])
def suggest(req: SuggestReq):
    words = get_suggestions(req.pattern)
    phrases = [f"{req.phrase} {s}" for s in words]

    # Calcular score BETO para cada frase
    scored_phrases = [(phrase, score_phrase(phrase)) for phrase in phrases]

    # Ordenar descendente por score (más probable primero)
    scored_phrases.sort(key=lambda x: x[1], reverse=True)

    recomendation = [p[0] for p in scored_phrases]

    return recomendation[0:5]