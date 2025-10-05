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
package com.sun.org.apache.xerces.internal.impl.dtd;

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
public class XMLContentSpec {

    @Positive
    public static final short CONTENTSPECNODE_LEAF;

    @Positive
    public static final short CONTENTSPECNODE_ZERO_OR_ONE;

    @Positive
    public static final short CONTENTSPECNODE_ZERO_OR_MORE;

    @Positive
    public static final short CONTENTSPECNODE_ONE_OR_MORE;

    @Positive
    public static final short CONTENTSPECNODE_CHOICE;

    @Positive
    public static final short CONTENTSPECNODE_SEQ;

    @Positive
    public static final short CONTENTSPECNODE_ANY;

    @Positive
    public static final short CONTENTSPECNODE_ANY_OTHER;

    @Positive
    public static final short CONTENTSPECNODE_ANY_LOCAL;

    @Positive
    public static final short CONTENTSPECNODE_ANY_LAX;

    @Positive
    public static final short CONTENTSPECNODE_ANY_OTHER_LAX;

    @Positive
    public static final short CONTENTSPECNODE_ANY_LOCAL_LAX;

    @Positive
    public static final short CONTENTSPECNODE_ANY_SKIP;

    @Positive
    public static final short CONTENTSPECNODE_ANY_OTHER_SKIP;

    @Positive
    public static final short CONTENTSPECNODE_ANY_LOCAL_SKIP;

    @Positive
    public short type;

    @Positive
    public Object value;

    @Positive
    public Object otherValue;

    @Positive
    public XMLContentSpec() {
    @Positive
    }

    @Positive
    public XMLContentSpec(short type, Object value, Object otherValue) {
    @Positive
    }

    @Positive
    public XMLContentSpec(XMLContentSpec contentSpec) {
    @Positive
    }

    @Positive
    public XMLContentSpec(XMLContentSpec.Provider provider, int contentSpecIndex) {
    @Positive
    }

    @Positive
    public void clear();

    @Positive
    public void setValues(short type, Object value, Object otherValue);

    @Positive
    public void setValues(XMLContentSpec contentSpec);

    @Positive
    public void setValues(XMLContentSpec.Provider provider, int contentSpecIndex);

    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object object);

    @Positive
    public interface Provider {

    @Positive
        public boolean getContentSpec(int contentSpecIndex, XMLContentSpec contentSpec);
    @Positive
    }
    @Positive
}
