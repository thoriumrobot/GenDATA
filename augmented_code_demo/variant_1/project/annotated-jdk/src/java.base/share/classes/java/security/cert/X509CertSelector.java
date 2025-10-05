/*
    @Positive
 * Copyright (c) 2000, 2020, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package java.security.cert;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.IOException;
    @Positive
import java.math.BigInteger;
    @Positive
import java.security.PublicKey;
    @Positive
import java.util.*;
    @Positive
import javax.security.auth.x500.X500Principal;
    @Positive
import sun.security.util.*;
    @Positive
import sun.security.x509.*;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class X509CertSelector implements CertSelector {

    @Positive
    public X509CertSelector() {
    @Positive
    }

    @Positive
    public void setCertificate(X509Certificate cert);

    @Positive
    public void setSerialNumber(BigInteger serial);

    @Positive
    public void setIssuer(X500Principal issuer);

    @Positive
    @Deprecated()
    @Positive
    public void setIssuer(String issuerDN) throws IOException;

    @Positive
    public void setIssuer(byte[] issuerDN) throws IOException;

    @Positive
    public void setSubject(X500Principal subject);

    @Positive
    @Deprecated()
    @Positive
    public void setSubject(String subjectDN) throws IOException;

    @Positive
    public void setSubject(byte[] subjectDN) throws IOException;

    @Positive
    public void setSubjectKeyIdentifier(byte[] subjectKeyID);

    @Positive
    public void setAuthorityKeyIdentifier(byte[] authorityKeyID);

    @Positive
    public void setCertificateValid(Date certValid);

    @Positive
    public void setPrivateKeyValid(Date privateKeyValid);

    @Positive
    public void setSubjectPublicKeyAlgID(String oid) throws IOException;

    @Positive
    public void setSubjectPublicKey(PublicKey key);

    @Positive
    public void setSubjectPublicKey(byte[] key) throws IOException;

    @Positive
    public void setKeyUsage(boolean[] keyUsage);

    @Positive
    public void setExtendedKeyUsage(Set<String> keyPurposeSet) throws IOException;

    @Positive
    public void setMatchAllSubjectAltNames(boolean matchAllNames);

    @Positive
    public void setSubjectAlternativeNames(Collection<List<?>> names) throws IOException;

    @Positive
    public void addSubjectAlternativeName(int type, String name) throws IOException;

    @Positive
    public void addSubjectAlternativeName(int type, byte[] name) throws IOException;

    @Positive
    static boolean equalNames(Collection<?> object1, Collection<?> object2);

    @Positive
    static GeneralNameInterface makeGeneralNameInterface(int type, Object name) throws IOException;

    @Positive
    public void setNameConstraints(byte[] bytes) throws IOException;

    @Positive
    public void setBasicConstraints(int minMaxPathLen);

    @Positive
    public void setPolicy(Set<String> certPolicySet) throws IOException;

    @Positive
    public void setPathToNames(Collection<List<?>> names) throws IOException;

    @Positive
    void setPathToNamesInternal(Set<GeneralNameInterface> names);

    @Positive
    public void addPathToName(int type, String name) throws IOException;

    @Positive
    public void addPathToName(int type, byte[] name) throws IOException;

    @Positive
    public X509Certificate getCertificate();

    @Positive
    public BigInteger getSerialNumber();

    @Positive
    public X500Principal getIssuer();

    @Positive
    @Deprecated()
    @Positive
    public String getIssuerAsString();

    @Positive
    public byte[] getIssuerAsBytes() throws IOException;

    @Positive
    public X500Principal getSubject();

    @Positive
    @Deprecated()
    @Positive
    public String getSubjectAsString();

    @Positive
    public byte[] getSubjectAsBytes() throws IOException;

    @Positive
    public byte[] getSubjectKeyIdentifier();

    @Positive
    public byte[] getAuthorityKeyIdentifier();

    @Positive
    public Date getCertificateValid();

    @Positive
    public Date getPrivateKeyValid();

    @Positive
    public String getSubjectPublicKeyAlgID();

    @Positive
    public PublicKey getSubjectPublicKey();

    @Positive
    public boolean[] getKeyUsage();

    @Positive
    public Set<String> getExtendedKeyUsage();

    @Positive
    public boolean getMatchAllSubjectAltNames();

    @Positive
    public Collection<List<?>> getSubjectAlternativeNames();

    @Positive
    public byte[] getNameConstraints();

    @Positive
    public int getBasicConstraints();

    @Positive
    public Set<String> getPolicy();

    @Positive
    public Collection<List<?>> getPathToNames();

    @Positive
    public String toString();

    @Positive
    public boolean match(Certificate cert);

    @Positive
    public Object clone();
    @Positive
}

// CFWR semantic augmentation - variant 1
