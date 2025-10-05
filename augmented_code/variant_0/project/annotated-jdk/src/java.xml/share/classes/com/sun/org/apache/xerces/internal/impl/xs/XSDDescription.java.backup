/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package com.sun.org.apache.xerces.internal.impl.xs;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import com.sun.org.apache.xerces.internal.util.XMLResourceIdentifierImpl;
import com.sun.org.apache.xerces.internal.xni.QName;
import com.sun.org.apache.xerces.internal.xni.XMLAttributes;
import com.sun.org.apache.xerces.internal.xni.grammars.XMLGrammarDescription;
import com.sun.org.apache.xerces.internal.xni.grammars.XMLSchemaDescription;

public class XSDDescription extends XMLResourceIdentifierImpl implements XMLSchemaDescription {

    public final static short CONTEXT_INITIALIZE;

    public final static short CONTEXT_INCLUDE;

    public final static short CONTEXT_REDEFINE;

    public final static short CONTEXT_IMPORT;

    public final static short CONTEXT_PREPARSE;

    public final static short CONTEXT_INSTANCE;

    public final static short CONTEXT_ELEMENT;

    public final static short CONTEXT_ATTRIBUTE;

    public final static short CONTEXT_XSITYPE;

    protected short fContextType;

    protected String[] fLocationHints;

    protected QName fTriggeringComponent;

    protected QName fEnclosedElementName;

    protected XMLAttributes fAttributes;

    public String getGrammarType();

    public short getContextType();

    public String getTargetNamespace();

    public String[] getLocationHints();

    public QName getTriggeringComponent();

    public QName getEnclosingElementName();

    public XMLAttributes getAttributes();

    public boolean fromInstance();

    public boolean isExternal();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object descObj);

    public int hashCode();

    public void setContextType(short contextType);

    public void setTargetNamespace(String targetNamespace);

    public void setLocationHints(String[] locationHints);

    public void setTriggeringComponent(QName triggeringComponent);

    public void setEnclosingElementName(QName enclosedElementName);

    public void setAttributes(XMLAttributes attributes);

    public void reset();

    public XSDDescription makeClone();
}
