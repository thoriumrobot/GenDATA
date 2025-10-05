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
import javax.xml.crypto.dom.*;
    @Positive
import javax.xml.crypto.dsig.*;
    @Positive
import javax.xml.crypto.dsig.dom.DOMSignContext;
    @Positive
import javax.xml.crypto.dsig.dom.DOMValidateContext;
    @Positive
import javax.xml.crypto.dsig.keyinfo.KeyInfo;
    @Positive
import java.security.InvalidKeyException;
    @Positive
import java.security.Key;
    @Positive
import java.security.Provider;
    @Positive
import java.util.Collections;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import org.w3c.dom.Attr;
    @Positive
import org.w3c.dom.Document;
    @Positive
import org.w3c.dom.Element;
    @Positive
import org.w3c.dom.Node;
    @Positive
import com.sun.org.apache.xml.internal.security.utils.XMLUtils;

    @Positive
public final class DOMXMLSignature extends DOMStructure implements XMLSignature {

    @Positive
    public DOMXMLSignature(SignedInfo si, KeyInfo ki, List<? extends XMLObject> objs, String id, String signatureValueId) {
    @Positive
    }

    @Positive
    public DOMXMLSignature(Element sigElem, XMLCryptoContext context, Provider provider) throws MarshalException {
    @Positive
    }

    @Positive
    public String getId();

    @Positive
    public KeyInfo getKeyInfo();

    @Positive
    public SignedInfo getSignedInfo();

    @Positive
    public List<XMLObject> getObjects();

    @Positive
    public SignatureValue getSignatureValue();

    @Positive
    public KeySelectorResult getKeySelectorResult();

    @Positive
    @Override
    @Positive
    public void marshal(Node parent, String dsPrefix, DOMCryptoContext context) throws MarshalException;

    @Positive
    public void marshal(Node parent, Node nextSibling, String dsPrefix, DOMCryptoContext context) throws MarshalException;

    @Positive
    @Override
    @Positive
    public boolean validate(XMLValidateContext vc) throws XMLSignatureException;

    @Positive
    @Override
    @Positive
    public void sign(XMLSignContext signContext) throws MarshalException, XMLSignatureException;

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
    public class DOMSignatureValue extends DOMStructure implements SignatureValue {

    @Positive
        public String getId();

    @Positive
        public byte[] getValue();

    @Positive
        public String getEncodedValue();

    @Positive
        @Override
    @Positive
        public boolean validate(XMLValidateContext validateContext) throws XMLSignatureException;

    @Positive
        @Override
    @Positive
        public boolean equals(Object o);

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        public void marshal(Node parent, String dsPrefix, DOMCryptoContext context) throws MarshalException;

    @Positive
        void setValue(byte[] value);
    @Positive
    }
    @Positive
}
