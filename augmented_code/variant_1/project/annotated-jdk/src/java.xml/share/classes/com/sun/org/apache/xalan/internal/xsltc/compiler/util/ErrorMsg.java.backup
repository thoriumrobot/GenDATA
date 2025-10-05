/*
 * Copyright (c) 2013, 2017, Oracle and/or its affiliates. All rights reserved.
 */
package com.sun.org.apache.xalan.internal.xsltc.compiler.util;

import org.checkerframework.checker.nullness.qual.Nullable;
import com.sun.org.apache.xalan.internal.xsltc.compiler.Stylesheet;
import com.sun.org.apache.xalan.internal.xsltc.compiler.SyntaxTreeNode;
import java.text.MessageFormat;
import java.util.Locale;
import java.util.ResourceBundle;
import jdk.xml.internal.SecuritySupport;

public final class ErrorMsg {

    public static final String MULTIPLE_STYLESHEET_ERR;

    public static final String TEMPLATE_REDEF_ERR;

    public static final String TEMPLATE_UNDEF_ERR;

    public static final String VARIABLE_REDEF_ERR;

    public static final String VARIABLE_UNDEF_ERR;

    public static final String CLASS_NOT_FOUND_ERR;

    public static final String METHOD_NOT_FOUND_ERR;

    public static final String ARGUMENT_CONVERSION_ERR;

    public static final String FILE_NOT_FOUND_ERR;

    public static final String INVALID_URI_ERR;

    public static final String CATALOG_EXCEPTION;

    public static final String FILE_ACCESS_ERR;

    public static final String MISSING_ROOT_ERR;

    public static final String NAMESPACE_UNDEF_ERR;

    public static final String FUNCTION_RESOLVE_ERR;

    public static final String NEED_LITERAL_ERR;

    public static final String XPATH_PARSER_ERR;

    public static final String REQUIRED_ATTR_ERR;

    public static final String ILLEGAL_CHAR_ERR;

    public static final String ILLEGAL_PI_ERR;

    public static final String STRAY_ATTRIBUTE_ERR;

    public static final String ILLEGAL_ATTRIBUTE_ERR;

    public static final String CIRCULAR_INCLUDE_ERR;

    public static final String IMPORT_PRECEDE_OTHERS_ERR;

    public static final String RESULT_TREE_SORT_ERR;

    public static final String SYMBOLS_REDEF_ERR;

    public static final String XSL_VERSION_ERR;

    public static final String CIRCULAR_VARIABLE_ERR;

    public static final String ILLEGAL_BINARY_OP_ERR;

    public static final String ILLEGAL_ARG_ERR;

    public static final String DOCUMENT_ARG_ERR;

    public static final String MISSING_WHEN_ERR;

    public static final String MULTIPLE_OTHERWISE_ERR;

    public static final String STRAY_OTHERWISE_ERR;

    public static final String STRAY_WHEN_ERR;

    public static final String WHEN_ELEMENT_ERR;

    public static final String UNNAMED_ATTRIBSET_ERR;

    public static final String ILLEGAL_CHILD_ERR;

    public static final String ILLEGAL_ELEM_NAME_ERR;

    public static final String ILLEGAL_ATTR_NAME_ERR;

    public static final String ILLEGAL_TEXT_NODE_ERR;

    public static final String SAX_PARSER_CONFIG_ERR;

    public static final String INTERNAL_ERR;

    public static final String UNSUPPORTED_XSL_ERR;

    public static final String UNSUPPORTED_EXT_ERR;

    public static final String MISSING_XSLT_URI_ERR;

    public static final String MISSING_XSLT_TARGET_ERR;

    public static final String ACCESSING_XSLT_TARGET_ERR;

    public static final String NOT_IMPLEMENTED_ERR;

    public static final String NOT_STYLESHEET_ERR;

    public static final String ELEMENT_PARSE_ERR;

    public static final String KEY_USE_ATTR_ERR;

    public static final String OUTPUT_VERSION_ERR;

    public static final String ILLEGAL_RELAT_OP_ERR;

    public static final String ATTRIBSET_UNDEF_ERR;

    public static final String ATTR_VAL_TEMPLATE_ERR;

    public static final String UNKNOWN_SIG_TYPE_ERR;

    public static final String DATA_CONVERSION_ERR;

    public static final String NO_TRANSLET_CLASS_ERR;

    public static final String NO_MAIN_TRANSLET_ERR;

    public static final String TRANSLET_CLASS_ERR;

    public static final String TRANSLET_OBJECT_ERR;

    public static final String ERROR_LISTENER_NULL_ERR;

    public static final String JAXP_UNKNOWN_SOURCE_ERR;

    public static final String JAXP_NO_SOURCE_ERR;

    public static final String JAXP_COMPILE_ERR;

    public static final String JAXP_INVALID_ATTR_ERR;

    public static final String JAXP_INVALID_ATTR_VALUE_ERR;

    public static final String JAXP_SET_RESULT_ERR;

    public static final String JAXP_NO_TRANSLET_ERR;

    public static final String JAXP_NO_HANDLER_ERR;

    public static final String JAXP_NO_RESULT_ERR;

    public static final String JAXP_UNKNOWN_PROP_ERR;

    public static final String SAX2DOM_ADAPTER_ERR;

    public static final String XSLTC_SOURCE_ERR;

    public static final String ER_RESULT_NULL;

    public static final String JAXP_INVALID_SET_PARAM_VALUE;

    public static final String JAXP_SET_FEATURE_NULL_NAME;

    public static final String JAXP_GET_FEATURE_NULL_NAME;

    public static final String JAXP_UNSUPPORTED_FEATURE;

    public static final String JAXP_SECUREPROCESSING_FEATURE;

    public static final String COMPILE_STDIN_ERR;

    public static final String COMPILE_USAGE_STR;

    public static final String TRANSFORM_USAGE_STR;

    public static final String STRAY_SORT_ERR;

    public static final String UNSUPPORTED_ENCODING;

    public static final String SYNTAX_ERR;

    public static final String CONSTRUCTOR_NOT_FOUND;

    public static final String NO_JAVA_FUNCT_THIS_REF;

    public static final String TYPE_CHECK_ERR;

    public static final String TYPE_CHECK_UNK_LOC_ERR;

    public static final String ILLEGAL_CMDLINE_OPTION_ERR;

    public static final String CMDLINE_OPT_MISSING_ARG_ERR;

    public static final String WARNING_PLUS_WRAPPED_MSG;

    public static final String WARNING_MSG;

    public static final String FATAL_ERR_PLUS_WRAPPED_MSG;

    public static final String FATAL_ERR_MSG;

    public static final String ERROR_PLUS_WRAPPED_MSG;

    public static final String ERROR_MSG;

    public static final String TRANSFORM_WITH_TRANSLET_STR;

    public static final String TRANSFORM_WITH_JAR_STR;

    public static final String COULD_NOT_CREATE_TRANS_FACT;

    public static final String TRANSLET_NAME_JAVA_CONFLICT;

    public static final String INVALID_QNAME_ERR;

    public static final String INVALID_NCNAME_ERR;

    public static final String INVALID_METHOD_IN_OUTPUT;

    public static final String OUTLINE_ERR_TRY_CATCH;

    public static final String OUTLINE_ERR_UNBALANCED_MARKERS;

    public static final String OUTLINE_ERR_DELETED_TARGET;

    public static final String OUTLINE_ERR_METHOD_TOO_BIG;

    public static final String DESERIALIZE_TRANSLET_ERR;

    public final static String ERROR_MESSAGES_KEY;

    public final static String COMPILER_ERROR_KEY;

    public final static String COMPILER_WARNING_KEY;

    public final static String RUNTIME_ERROR_KEY;

    public ErrorMsg(String code) {
    }

    public ErrorMsg(String code, Throwable e) {
    }

    public ErrorMsg(String message, int line) {
    }

    public ErrorMsg(String code, int line, Object param) {
    }

    public ErrorMsg(String code, Object param) {
    }

    public ErrorMsg(String code, Object param1, Object param2) {
    }

    public ErrorMsg(String code, SyntaxTreeNode node) {
    }

    public ErrorMsg(String code, Object param1, SyntaxTreeNode node) {
    }

    public ErrorMsg(String code, Object param1, Object param2, SyntaxTreeNode node) {
    }

    @Nullable
    public Throwable getCause();

    public String toString();

    public String toString(Object obj);

    public String toString(Object obj0, Object obj1);

    public void setWarningError(boolean flag);

    public boolean isWarningError();
}
