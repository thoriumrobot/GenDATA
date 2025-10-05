/*
    @Positive
 * reserved comment block
    @Positive
 * DO NOT REMOVE OR ALTER!
    @Positive
 */
    @Positive
package org.jcp.xml.dsig.internal.dom;

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
import javax.xml.crypto.*;
    @Positive
import javax.xml.crypto.dom.DOMCryptoContext;
    @Positive
import javax.xml.crypto.dsig.*;
    @Positive
import java.util.*;
    @Positive
import org.w3c.dom.Attr;
    @Positive
import org.w3c.dom.Document;
    @Positive
import org.w3c.dom.Element;
    @Positive
import org.w3c.dom.Node;

    @Positive
public final class DOMSignatureProperty extends DOMStructure implements SignatureProperty {

    @Positive
    public DOMSignatureProperty(List<? extends XMLStructure> content, String target, String id) {
    @Positive
    }

    @Positive
    public DOMSignatureProperty(Element propElem) throws MarshalException {
    @Positive
    }

    @Positive
    public List<XMLStructure> getContent();

    @Positive
    public String getId();

    @Positive
    public String getTarget();

    @Positive
    @Override
    @Positive
    public void marshal(Node parent, String dsPrefix, DOMCryptoContext context) throws MarshalException;

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    @Override
    @Positive
    public int hashCode();
    @Positive
}

// CFWR semantic augmentation - variant 1
