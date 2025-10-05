/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * reserved comment block
    @Positive
 * DO NOT REMOVE OR ALTER!
    @Positive
 */
    @Positive
package com.sun.org.apache.xerces.internal.impl.xs;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import com.sun.org.apache.xerces.internal.util.XMLResourceIdentifierImpl;
    @Positive
import com.sun.org.apache.xerces.internal.xni.QName;
    @Positive
import com.sun.org.apache.xerces.internal.xni.XMLAttributes;
    @Positive
import com.sun.org.apache.xerces.internal.xni.grammars.XMLGrammarDescription;
    @Positive
import com.sun.org.apache.xerces.internal.xni.grammars.XMLSchemaDescription;

    @Positive
public class XSDDescription extends XMLResourceIdentifierImpl implements XMLSchemaDescription {

    @Positive
    public final static short CONTEXT_INITIALIZE;

    @Positive
    public final static short CONTEXT_INCLUDE;

    @Positive
    public final static short CONTEXT_REDEFINE;

    @Positive
    public final static short CONTEXT_IMPORT;

    @Positive
    public final static short CONTEXT_PREPARSE;

    @Positive
    public final static short CONTEXT_INSTANCE;

    @Positive
    public final static short CONTEXT_ELEMENT;

    @Positive
    public final static short CONTEXT_ATTRIBUTE;

    @Positive
    public final static short CONTEXT_XSITYPE;

    @Positive
    protected short fContextType;

    @Positive
    protected String[] fLocationHints;

    @Positive
    protected QName fTriggeringComponent;

    @Positive
    protected QName fEnclosedElementName;

    @Positive
    protected XMLAttributes fAttributes;

    @Positive
    public String getGrammarType();

    @Positive
    public short getContextType();

    @Positive
    public String getTargetNamespace();

    @Positive
    public String[] getLocationHints();

    @Positive
    public QName getTriggeringComponent();

    @Positive
    public QName getEnclosingElementName();

    @Positive
    public XMLAttributes getAttributes();

    @Positive
    public boolean fromInstance();

    @Positive
    public boolean isExternal();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object descObj);

    @Positive
    public int hashCode();

    @Positive
    public void setContextType(short contextType);

    @Positive
    public void setTargetNamespace(String targetNamespace);

    @Positive
    public void setLocationHints(String[] locationHints);

    @Positive
    public void setTriggeringComponent(QName triggeringComponent);

    @Positive
    public void setEnclosingElementName(QName enclosedElementName);

    @Positive
    public void setAttributes(XMLAttributes attributes);

    @Positive
    public void reset();

    @Positive
    public XSDDescription makeClone();
    @Positive
}
