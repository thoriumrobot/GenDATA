/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package org.jcp.xml.dsig.internal.dom;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.security.Provider;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import javax.xml.crypto.MarshalException;
import javax.xml.crypto.XMLCryptoContext;
import javax.xml.crypto.XMLStructure;
import javax.xml.crypto.dom.DOMCryptoContext;
import javax.xml.crypto.dsig.XMLSignature;
import javax.xml.crypto.dsig.dom.DOMSignContext;
import javax.xml.crypto.dsig.keyinfo.KeyInfo;
import org.w3c.dom.Attr;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;

public final class DOMKeyInfo extends DOMStructure implements KeyInfo {

    @SuppressWarnings("unchecked")
    public static List<XMLStructure> getContent(KeyInfo ki);

    public DOMKeyInfo(List<? extends XMLStructure> content, String id) {
    }

    public DOMKeyInfo(Element kiElem, XMLCryptoContext context, Provider provider) throws MarshalException {
    }

    public String getId();

    public List<XMLStructure> getContent();

    public void marshal(XMLStructure parent, XMLCryptoContext context) throws MarshalException;

    @Override
    public void marshal(Node parent, String dsPrefix, DOMCryptoContext context) throws MarshalException;

    public void marshal(Node parent, Node nextSibling, String dsPrefix, DOMCryptoContext context) throws MarshalException;

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    @Override
    public int hashCode();
}
