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
import javax.xml.crypto.*;
import javax.xml.crypto.dom.*;
import javax.xml.crypto.dsig.*;
import javax.xml.crypto.dsig.dom.DOMSignContext;
import javax.xml.crypto.dsig.dom.DOMValidateContext;
import javax.xml.crypto.dsig.keyinfo.KeyInfo;
import java.security.InvalidKeyException;
import java.security.Key;
import java.security.Provider;
import java.util.Collections;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.w3c.dom.Attr;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import com.sun.org.apache.xml.internal.security.utils.XMLUtils;

public final class DOMXMLSignature extends DOMStructure implements XMLSignature {

    public DOMXMLSignature(SignedInfo si, KeyInfo ki, List<? extends XMLObject> objs, String id, String signatureValueId) {
    }

    public DOMXMLSignature(Element sigElem, XMLCryptoContext context, Provider provider) throws MarshalException {
    }

    public String getId();

    public KeyInfo getKeyInfo();

    public SignedInfo getSignedInfo();

    public List<XMLObject> getObjects();

    public SignatureValue getSignatureValue();

    public KeySelectorResult getKeySelectorResult();

    @Override
    public void marshal(Node parent, String dsPrefix, DOMCryptoContext context) throws MarshalException;

    public void marshal(Node parent, Node nextSibling, String dsPrefix, DOMCryptoContext context) throws MarshalException;

    @Override
    public boolean validate(XMLValidateContext vc) throws XMLSignatureException;

    @Override
    public void sign(XMLSignContext signContext) throws MarshalException, XMLSignatureException;

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    @Override
    public int hashCode();

    public class DOMSignatureValue extends DOMStructure implements SignatureValue {

        public String getId();

        public byte[] getValue();

        public String getEncodedValue();

        @Override
        public boolean validate(XMLValidateContext validateContext) throws XMLSignatureException;

        @Override
        public boolean equals(Object o);

        @Override
        public int hashCode();

        public void marshal(Node parent, String dsPrefix, DOMCryptoContext context) throws MarshalException;

        void setValue(byte[] value);
    }
}
