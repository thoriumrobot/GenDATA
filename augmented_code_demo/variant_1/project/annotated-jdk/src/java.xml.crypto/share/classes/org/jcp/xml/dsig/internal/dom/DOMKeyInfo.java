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
import java.security.Provider;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collections;
    @Positive
import java.util.List;
    @Positive
import javax.xml.crypto.MarshalException;
    @Positive
import javax.xml.crypto.XMLCryptoContext;
    @Positive
import javax.xml.crypto.XMLStructure;
    @Positive
import javax.xml.crypto.dom.DOMCryptoContext;
    @Positive
import javax.xml.crypto.dsig.XMLSignature;
    @Positive
import javax.xml.crypto.dsig.dom.DOMSignContext;
    @Positive
import javax.xml.crypto.dsig.keyinfo.KeyInfo;
    @Positive
import org.w3c.dom.Attr;
    @Positive
import org.w3c.dom.Document;
    @Positive
import org.w3c.dom.Element;
    @Positive
import org.w3c.dom.Node;

    @Positive
public final class DOMKeyInfo extends DOMStructure implements KeyInfo {

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public static List<XMLStructure> getContent(KeyInfo ki);

    @Positive
    public DOMKeyInfo(List<? extends XMLStructure> content, String id) {
    @Positive
    }

    @Positive
    public DOMKeyInfo(Element kiElem, XMLCryptoContext context, Provider provider) throws MarshalException {
    @Positive
    }

    @Positive
    public String getId();

    @Positive
    public List<XMLStructure> getContent();

    @Positive
    public void marshal(XMLStructure parent, XMLCryptoContext context) throws MarshalException;

    @Positive
    @Override
    @Positive
    public void marshal(Node parent, String dsPrefix, DOMCryptoContext context) throws MarshalException;

    @Positive
    public void marshal(Node parent, Node nextSibling, String dsPrefix, DOMCryptoContext context) throws MarshalException;

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
