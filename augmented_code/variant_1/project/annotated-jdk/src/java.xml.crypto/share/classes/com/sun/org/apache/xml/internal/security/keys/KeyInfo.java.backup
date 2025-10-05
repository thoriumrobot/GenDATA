/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package com.sun.org.apache.xml.internal.security.keys;

import org.checkerframework.dataflow.qual.Pure;
import java.security.PrivateKey;
import java.security.PublicKey;
import java.security.cert.X509Certificate;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import javax.crypto.SecretKey;
import com.sun.org.apache.xml.internal.security.exceptions.XMLSecurityException;
import com.sun.org.apache.xml.internal.security.keys.content.DEREncodedKeyValue;
import com.sun.org.apache.xml.internal.security.keys.content.KeyInfoReference;
import com.sun.org.apache.xml.internal.security.keys.content.KeyName;
import com.sun.org.apache.xml.internal.security.keys.content.KeyValue;
import com.sun.org.apache.xml.internal.security.keys.content.MgmtData;
import com.sun.org.apache.xml.internal.security.keys.content.PGPData;
import com.sun.org.apache.xml.internal.security.keys.content.RetrievalMethod;
import com.sun.org.apache.xml.internal.security.keys.content.SPKIData;
import com.sun.org.apache.xml.internal.security.keys.content.X509Data;
import com.sun.org.apache.xml.internal.security.keys.content.keyvalues.DSAKeyValue;
import com.sun.org.apache.xml.internal.security.keys.content.keyvalues.RSAKeyValue;
import com.sun.org.apache.xml.internal.security.keys.keyresolver.KeyResolver;
import com.sun.org.apache.xml.internal.security.keys.keyresolver.KeyResolverException;
import com.sun.org.apache.xml.internal.security.keys.keyresolver.KeyResolverSpi;
import com.sun.org.apache.xml.internal.security.keys.storage.StorageResolver;
import com.sun.org.apache.xml.internal.security.transforms.Transforms;
import com.sun.org.apache.xml.internal.security.utils.Constants;
import com.sun.org.apache.xml.internal.security.utils.ElementProxy;
import com.sun.org.apache.xml.internal.security.utils.SignatureElementProxy;
import com.sun.org.apache.xml.internal.security.utils.XMLUtils;
import org.w3c.dom.Attr;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;

public class KeyInfo extends SignatureElementProxy {

    public KeyInfo(Document doc) {
    }

    public KeyInfo(Element element, String baseURI) throws XMLSecurityException {
    }

    public void setSecureValidation(boolean secureValidation);

    public void setId(String id);

    public String getId();

    public void addKeyName(String keynameString);

    public void add(KeyName keyname);

    public void addKeyValue(PublicKey pk);

    public void addKeyValue(Element unknownKeyValueElement);

    public void add(DSAKeyValue dsakeyvalue);

    public void add(RSAKeyValue rsakeyvalue);

    public void add(PublicKey pk);

    public void add(KeyValue keyvalue);

    public void addMgmtData(String mgmtdata);

    public void add(MgmtData mgmtdata);

    public void add(PGPData pgpdata);

    public void addRetrievalMethod(String uri, Transforms transforms, String Type);

    public void add(RetrievalMethod retrievalmethod);

    public void add(SPKIData spkidata);

    public void add(X509Data x509data);

    public void addDEREncodedKeyValue(PublicKey pk) throws XMLSecurityException;

    public void add(DEREncodedKeyValue derEncodedKeyValue);

    public void addKeyInfoReference(String URI) throws XMLSecurityException;

    public void add(KeyInfoReference keyInfoReference);

    public void addUnknownElement(Element element);

    public int lengthKeyName();

    public int lengthKeyValue();

    public int lengthMgmtData();

    public int lengthPGPData();

    public int lengthRetrievalMethod();

    public int lengthSPKIData();

    public int lengthX509Data();

    public int lengthDEREncodedKeyValue();

    public int lengthKeyInfoReference();

    public int lengthUnknownElement();

    public KeyName itemKeyName(int i) throws XMLSecurityException;

    public KeyValue itemKeyValue(int i) throws XMLSecurityException;

    public MgmtData itemMgmtData(int i) throws XMLSecurityException;

    public PGPData itemPGPData(int i) throws XMLSecurityException;

    public RetrievalMethod itemRetrievalMethod(int i) throws XMLSecurityException;

    public SPKIData itemSPKIData(int i) throws XMLSecurityException;

    public X509Data itemX509Data(int i) throws XMLSecurityException;

    public DEREncodedKeyValue itemDEREncodedKeyValue(int i) throws XMLSecurityException;

    public KeyInfoReference itemKeyInfoReference(int i) throws XMLSecurityException;

    public Element itemUnknownElement(int i);

    public boolean isEmpty();

    @Pure
    public boolean containsKeyName();

    @Pure
    public boolean containsKeyValue();

    @Pure
    public boolean containsMgmtData();

    @Pure
    public boolean containsPGPData();

    @Pure
    public boolean containsRetrievalMethod();

    @Pure
    public boolean containsSPKIData();

    @Pure
    public boolean containsUnknownElement();

    @Pure
    public boolean containsX509Data();

    @Pure
    public boolean containsDEREncodedKeyValue();

    @Pure
    public boolean containsKeyInfoReference();

    public PublicKey getPublicKey() throws KeyResolverException;

    PublicKey getPublicKeyFromStaticResolvers() throws KeyResolverException;

    PublicKey getPublicKeyFromInternalResolvers() throws KeyResolverException;

    public X509Certificate getX509Certificate() throws KeyResolverException;

    X509Certificate getX509CertificateFromStaticResolvers() throws KeyResolverException;

    X509Certificate getX509CertificateFromInternalResolvers() throws KeyResolverException;

    public SecretKey getSecretKey() throws KeyResolverException;

    SecretKey getSecretKeyFromStaticResolvers() throws KeyResolverException;

    SecretKey getSecretKeyFromInternalResolvers() throws KeyResolverException;

    public PrivateKey getPrivateKey() throws KeyResolverException;

    PrivateKey getPrivateKeyFromStaticResolvers() throws KeyResolverException;

    PrivateKey getPrivateKeyFromInternalResolvers() throws KeyResolverException;

    public void registerInternalKeyResolver(KeyResolverSpi realKeyResolver);

    int lengthInternalKeyResolver();

    KeyResolverSpi itemInternalKeyResolver(int i);

    public void addStorageResolver(StorageResolver storageResolver);

    public String getBaseLocalName();
}
