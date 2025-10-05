/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2013, 2017, Oracle and/or its affiliates. All rights reserved.
    @Positive
 */
    @Positive
package com.sun.org.apache.xalan.internal.xsltc.compiler.util;

    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import com.sun.org.apache.xalan.internal.xsltc.compiler.Stylesheet;
    @Positive
import com.sun.org.apache.xalan.internal.xsltc.compiler.SyntaxTreeNode;
    @Positive
import java.text.MessageFormat;
    @Positive
import java.util.Locale;
    @Positive
import java.util.ResourceBundle;
    @Positive
import jdk.xml.internal.SecuritySupport;

    @Positive
public final class ErrorMsg {

    @Positive
    public static final String MULTIPLE_STYLESHEET_ERR;

    @Positive
    public static final String TEMPLATE_REDEF_ERR;

    @Positive
    public static final String TEMPLATE_UNDEF_ERR;

    @Positive
    public static final String VARIABLE_REDEF_ERR;

    @Positive
    public static final String VARIABLE_UNDEF_ERR;

    @Positive
    public static final String CLASS_NOT_FOUND_ERR;

    @Positive
    public static final String METHOD_NOT_FOUND_ERR;

    @Positive
    public static final String ARGUMENT_CONVERSION_ERR;

    @Positive
    public static final String FILE_NOT_FOUND_ERR;

    @Positive
    public static final String INVALID_URI_ERR;

    @Positive
    public static final String CATALOG_EXCEPTION;

    @Positive
    public static final String FILE_ACCESS_ERR;

    @Positive
    public static final String MISSING_ROOT_ERR;

    @Positive
    public static final String NAMESPACE_UNDEF_ERR;

    @Positive
    public static final String FUNCTION_RESOLVE_ERR;

    @Positive
    public static final String NEED_LITERAL_ERR;

    @Positive
    public static final String XPATH_PARSER_ERR;

    @Positive
    public static final String REQUIRED_ATTR_ERR;

    @Positive
    public static final String ILLEGAL_CHAR_ERR;

    @Positive
    public static final String ILLEGAL_PI_ERR;

    @Positive
    public static final String STRAY_ATTRIBUTE_ERR;

    @Positive
    public static final String ILLEGAL_ATTRIBUTE_ERR;

    @Positive
    public static final String CIRCULAR_INCLUDE_ERR;

    @Positive
    public static final String IMPORT_PRECEDE_OTHERS_ERR;

    @Positive
    public static final String RESULT_TREE_SORT_ERR;

    @Positive
    public static final String SYMBOLS_REDEF_ERR;

    @Positive
    public static final String XSL_VERSION_ERR;

    @Positive
    public static final String CIRCULAR_VARIABLE_ERR;

    @Positive
    public static final String ILLEGAL_BINARY_OP_ERR;

    @Positive
    public static final String ILLEGAL_ARG_ERR;

    @Positive
    public static final String DOCUMENT_ARG_ERR;

    @Positive
    public static final String MISSING_WHEN_ERR;

    @Positive
    public static final String MULTIPLE_OTHERWISE_ERR;

    @Positive
    public static final String STRAY_OTHERWISE_ERR;

    @Positive
    public static final String STRAY_WHEN_ERR;

    @Positive
    public static final String WHEN_ELEMENT_ERR;

    @Positive
    public static final String UNNAMED_ATTRIBSET_ERR;

    @Positive
    public static final String ILLEGAL_CHILD_ERR;

    @Positive
    public static final String ILLEGAL_ELEM_NAME_ERR;

    @Positive
    public static final String ILLEGAL_ATTR_NAME_ERR;

    @Positive
    public static final String ILLEGAL_TEXT_NODE_ERR;

    @Positive
    public static final String SAX_PARSER_CONFIG_ERR;

    @Positive
    public static final String INTERNAL_ERR;

    @Positive
    public static final String UNSUPPORTED_XSL_ERR;

    @Positive
    public static final String UNSUPPORTED_EXT_ERR;

    @Positive
    public static final String MISSING_XSLT_URI_ERR;

    @Positive
    public static final String MISSING_XSLT_TARGET_ERR;

    @Positive
    public static final String ACCESSING_XSLT_TARGET_ERR;

    @Positive
    public static final String NOT_IMPLEMENTED_ERR;

    @Positive
    public static final String NOT_STYLESHEET_ERR;

    @Positive
    public static final String ELEMENT_PARSE_ERR;

    @Positive
    public static final String KEY_USE_ATTR_ERR;

    @Positive
    public static final String OUTPUT_VERSION_ERR;

    @Positive
    public static final String ILLEGAL_RELAT_OP_ERR;

    @Positive
    public static final String ATTRIBSET_UNDEF_ERR;

    @Positive
    public static final String ATTR_VAL_TEMPLATE_ERR;

    @Positive
    public static final String UNKNOWN_SIG_TYPE_ERR;

    @Positive
    public static final String DATA_CONVERSION_ERR;

    @Positive
    public static final String NO_TRANSLET_CLASS_ERR;

    @Positive
    public static final String NO_MAIN_TRANSLET_ERR;

    @Positive
    public static final String TRANSLET_CLASS_ERR;

    @Positive
    public static final String TRANSLET_OBJECT_ERR;

    @Positive
    public static final String ERROR_LISTENER_NULL_ERR;

    @Positive
    public static final String JAXP_UNKNOWN_SOURCE_ERR;

    @Positive
    public static final String JAXP_NO_SOURCE_ERR;

    @Positive
    public static final String JAXP_COMPILE_ERR;

    @Positive
    public static final String JAXP_INVALID_ATTR_ERR;

    @Positive
    public static final String JAXP_INVALID_ATTR_VALUE_ERR;

    @Positive
    public static final String JAXP_SET_RESULT_ERR;

    @Positive
    public static final String JAXP_NO_TRANSLET_ERR;

    @Positive
    public static final String JAXP_NO_HANDLER_ERR;

    @Positive
    public static final String JAXP_NO_RESULT_ERR;

    @Positive
    public static final String JAXP_UNKNOWN_PROP_ERR;

    @Positive
    public static final String SAX2DOM_ADAPTER_ERR;

    @Positive
    public static final String XSLTC_SOURCE_ERR;

    @Positive
    public static final String ER_RESULT_NULL;

    @Positive
    public static final String JAXP_INVALID_SET_PARAM_VALUE;

    @Positive
    public static final String JAXP_SET_FEATURE_NULL_NAME;

    @Positive
    public static final String JAXP_GET_FEATURE_NULL_NAME;

    @Positive
    public static final String JAXP_UNSUPPORTED_FEATURE;

    @Positive
    public static final String JAXP_SECUREPROCESSING_FEATURE;

    @Positive
    public static final String COMPILE_STDIN_ERR;

    @Positive
    public static final String COMPILE_USAGE_STR;

    @Positive
    public static final String TRANSFORM_USAGE_STR;

    @Positive
    public static final String STRAY_SORT_ERR;

    @Positive
    public static final String UNSUPPORTED_ENCODING;

    @Positive
    public static final String SYNTAX_ERR;

    @Positive
    public static final String CONSTRUCTOR_NOT_FOUND;

    @Positive
    public static final String NO_JAVA_FUNCT_THIS_REF;

    @Positive
    public static final String TYPE_CHECK_ERR;

    @Positive
    public static final String TYPE_CHECK_UNK_LOC_ERR;

    @Positive
    public static final String ILLEGAL_CMDLINE_OPTION_ERR;

    @Positive
    public static final String CMDLINE_OPT_MISSING_ARG_ERR;

    @Positive
    public static final String WARNING_PLUS_WRAPPED_MSG;

    @Positive
    public static final String WARNING_MSG;

    @Positive
    public static final String FATAL_ERR_PLUS_WRAPPED_MSG;

    @Positive
    public static final String FATAL_ERR_MSG;

    @Positive
    public static final String ERROR_PLUS_WRAPPED_MSG;

    @Positive
    public static final String ERROR_MSG;

    @Positive
    public static final String TRANSFORM_WITH_TRANSLET_STR;

    @Positive
    public static final String TRANSFORM_WITH_JAR_STR;

    @Positive
    public static final String COULD_NOT_CREATE_TRANS_FACT;

    @Positive
    public static final String TRANSLET_NAME_JAVA_CONFLICT;

    @Positive
    public static final String INVALID_QNAME_ERR;

    @Positive
    public static final String INVALID_NCNAME_ERR;

    @Positive
    public static final String INVALID_METHOD_IN_OUTPUT;

    @Positive
    public static final String OUTLINE_ERR_TRY_CATCH;

    @Positive
    public static final String OUTLINE_ERR_UNBALANCED_MARKERS;

    @Positive
    public static final String OUTLINE_ERR_DELETED_TARGET;

    @Positive
    public static final String OUTLINE_ERR_METHOD_TOO_BIG;

    @Positive
    public static final String DESERIALIZE_TRANSLET_ERR;

    @Positive
    public final static String ERROR_MESSAGES_KEY;

    @Positive
    public final static String COMPILER_ERROR_KEY;

    @Positive
    public final static String COMPILER_WARNING_KEY;

    @Positive
    public final static String RUNTIME_ERROR_KEY;

    @Positive
    public ErrorMsg(String code) {
    @Positive
    }

    @Positive
    public ErrorMsg(String code, Throwable e) {
    @Positive
    }

    @Positive
    public ErrorMsg(String message, int line) {
    @Positive
    }

    @Positive
    public ErrorMsg(String code, int line, Object param) {
    @Positive
    }

    @Positive
    public ErrorMsg(String code, Object param) {
    @Positive
    }

    @Positive
    public ErrorMsg(String code, Object param1, Object param2) {
    @Positive
    }

    @Positive
    public ErrorMsg(String code, SyntaxTreeNode node) {
    @Positive
    }

    @Positive
    public ErrorMsg(String code, Object param1, SyntaxTreeNode node) {
    @Positive
    }

    @Positive
    public ErrorMsg(String code, Object param1, Object param2, SyntaxTreeNode node) {
    @Positive
    }

    @Positive
    @Nullable
    @Positive
    public Throwable getCause();

    @Positive
    public String toString();

    @Positive
    public String toString(Object obj);

    @Positive
    public String toString(Object obj0, Object obj1);

    @Positive
    public void setWarningError(boolean flag);

    @Positive
    public boolean isWarningError();
    @Positive
}
