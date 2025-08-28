
import os, ast, math, statistics, signal, logging
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel
from typing import Optional

# Configurar logging estructurado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI()
TIMEOUT = int(os.getenv("PYEXEC_TIMEOUT_SEC", "5"))

# Configuración adicional de validación
MAX_EXPR_LENGTH = int(os.getenv("MAX_EXPR_LENGTH", "500"))
MAX_EXPR_COMPLEXITY = int(os.getenv("MAX_EXPR_COMPLEXITY", "100"))  # Número máximo de nodos AST

# Categorías de errores
class ErrorCategory:
    SYNTAX_ERROR = "syntax_error"
    TIMEOUT = "timeout"
    FORBIDDEN_FUNCTION = "forbidden_function"
    FORBIDDEN_NODE = "forbidden_node"
    TOO_LONG = "too_long"
    TOO_COMPLEX = "too_complex"
    RUNTIME_ERROR = "runtime_error"

SAFE_GLOBALS = {"__builtins__": {}}
ALLOWED = {}
for mod in (math, statistics):
    for name in dir(mod):
        if not name.startswith("_"):
            ALLOWED[name] = getattr(mod, name)
ALLOWED.update({"abs": abs, "min": min, "max": max, "sum": sum, "len": len, "round": round, "pow": pow, "divmod": divmod, "bin": bin, "hex": hex, "oct": oct, "int": int, "float": float, "complex": complex, "bool": bool, "range": range, "enumerate": enumerate, "zip": zip, "sorted": sorted, "reversed": reversed})
SAFE_GLOBALS.update(ALLOWED)

ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.List, ast.Tuple, ast.Dict,
    ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.Call, ast.Name, ast.Constant, ast.Compare, ast.Eq, ast.NotEq,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.BoolOp, ast.And, ast.Or
)

class ValidationError(Exception):
    """Error personalizado de validación con categoría"""
    def __init__(self, message: str, category: str):
        super().__init__(message)
        self.category = category

def validate_expression_length(expr: str):
    """Valida la longitud de la expresión"""
    if len(expr) > MAX_EXPR_LENGTH:
        raise ValidationError(
            f"La expresión es demasiado larga (máximo {MAX_EXPR_LENGTH} caracteres)",
            ErrorCategory.TOO_LONG
        )

def validate_expression_complexity(parsed_ast):
    """Valida la complejidad de la expresión contando nodos AST"""
    node_count = sum(1 for _ in ast.walk(parsed_ast))
    if node_count > MAX_EXPR_COMPLEXITY:
        raise ValidationError(
            f"La expresión es demasiado compleja (máximo {MAX_EXPR_COMPLEXITY} operaciones)",
            ErrorCategory.TOO_COMPLEX
        )

def validate_ast(expr: str):
    logger.info(f"Validating AST for expression - length: {len(expr)}")
    
    # Validar longitud
    validate_expression_length(expr)
    
    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        logger.warning(f"Syntax error in expression: {str(e)}")
        raise ValidationError(
            "Error de sintaxis en la expresión matemática",
            ErrorCategory.SYNTAX_ERROR
        )
    
    # Validar complejidad
    validate_expression_complexity(parsed)
    
    # Validar nodos y funciones permitidas
    for node in ast.walk(parsed):
        if not isinstance(node, ALLOWED_NODES):
            logger.warning(f"Blocked node type: {type(node).__name__}")
            raise ValidationError(
                f"Operación no permitida: {type(node).__name__}",
                ErrorCategory.FORBIDDEN_NODE
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED:
                func_name = node.func.id if hasattr(node.func, 'id') else 'unknown'
                logger.warning(f"Blocked function call: {func_name}")
                raise ValidationError(
                    f"Función no permitida: {func_name}",
                    ErrorCategory.FORBIDDEN_FUNCTION
                )
    
    logger.info("AST validation successful")
    return parsed

class CodeIn(BaseModel):
    expr: str

def alarm_handler(signum, frame):
    raise TimeoutError(f"Tiempo de ejecución excedido ({TIMEOUT} segundos)")

@app.get("/health")
def health():
    logger.info("Health check requested")
    return {"status": "ok"}

def create_safe_error_response(error: Exception, category: str) -> dict:
    """Crea una respuesta de error segura sin exponer detalles internos"""
    error_messages = {
        ErrorCategory.SYNTAX_ERROR: "Error de sintaxis en la expresión matemática. Verifica que la operación esté bien formada.",
        ErrorCategory.TIMEOUT: f"La operación tardó demasiado tiempo (máximo {TIMEOUT} segundos). Intenta con una operación más simple.",
        ErrorCategory.FORBIDDEN_FUNCTION: "Se intentó usar una función no permitida por seguridad. Solo se permiten operaciones matemáticas básicas.",
        ErrorCategory.FORBIDDEN_NODE: "Se intentó usar una operación no permitida por seguridad. Solo se permiten cálculos matemáticos.",
        ErrorCategory.TOO_LONG: f"La expresión es demasiado larga (máximo {MAX_EXPR_LENGTH} caracteres). Intenta simplificar la operación.",
        ErrorCategory.TOO_COMPLEX: f"La expresión es demasiado compleja (máximo {MAX_EXPR_COMPLEXITY} operaciones). Divide el cálculo en pasos más pequeños.",
        ErrorCategory.RUNTIME_ERROR: "Error durante la ejecución de la expresión. Verifica que los números y operaciones sean válidos."
    }
    
    user_message = error_messages.get(category, "Error al procesar la expresión")
    
    return {
        "ok": False,
        "error": {
            "category": category,
            "message": user_message
        }
    }

@app.post("/run")
def run_code(body: CodeIn, x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID")):
    correlation_id = x_correlation_id or "unknown"
    logger.info(f"Code execution request received - correlation_id: {correlation_id}, expr_length: {len(body.expr)}")
    
    try:
        # Validación completa de la expresión
        parsed = validate_ast(body.expr)
        
        # Configurar timeout
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(TIMEOUT)
        
        logger.info(f"Executing Python expression - correlation_id: {correlation_id}")
        result = eval(compile(parsed, "<expr>", "eval"), SAFE_GLOBALS, {})
        signal.alarm(0)  # Cancelar alarm si se completó exitosamente
        
        logger.info(f"Code execution successful - correlation_id: {correlation_id}, result_type: {type(result).__name__}")
        return {"ok": True, "result": result}
        
    except ValidationError as e:
        logger.warning(f"Validation failed - correlation_id: {correlation_id}, category: {e.category}, message: {str(e)}")
        response = create_safe_error_response(e, e.category)
        raise HTTPException(status_code=400, detail=response)
        
    except TimeoutError as e:
        logger.warning(f"Code execution timeout - correlation_id: {correlation_id}, timeout: {TIMEOUT}s")
        signal.alarm(0)  # Limpiar alarm
        response = create_safe_error_response(e, ErrorCategory.TIMEOUT)
        raise HTTPException(status_code=400, detail=response)
        
    except ZeroDivisionError as e:
        logger.warning(f"Division by zero error - correlation_id: {correlation_id}")
        response = create_safe_error_response(e, ErrorCategory.RUNTIME_ERROR)
        response["error"]["message"] = "División por cero detectada. No se puede dividir entre cero."
        raise HTTPException(status_code=400, detail=response)
        
    except ValueError as e:
        logger.warning(f"Value error during execution - correlation_id: {correlation_id}, error: {str(e)}")
        response = create_safe_error_response(e, ErrorCategory.RUNTIME_ERROR)
        response["error"]["message"] = "Error en los valores de la operación. Verifica que todos los números sean válidos para esta operación matemática."
        raise HTTPException(status_code=400, detail=response)
        
    except Exception as e:
        logger.error(f"Unexpected error during code execution - correlation_id: {correlation_id}, error: {str(e)}")
        response = create_safe_error_response(e, ErrorCategory.RUNTIME_ERROR)
        raise HTTPException(status_code=500, detail=response)
