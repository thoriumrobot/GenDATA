/*
 * Copyright (c) 2000, 2020, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package java.security.cert;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.IOException;
import java.math.BigInteger;
import java.security.PublicKey;
import java.util.*;
import javax.security.auth.x500.X500Principal;
import sun.security.util.*;
import sun.security.x509.*;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class X509CertSelector implements CertSelector {

    public X509CertSelector() {
    }

    public void setCertificate(X509Certificate cert);

    public void setSerialNumber(BigInteger serial);

    public void setIssuer(X500Principal issuer);

    @Deprecated()
    public void setIssuer(String issuerDN) throws IOException;

    public void setIssuer(byte[] issuerDN) throws IOException;

    public void setSubject(X500Principal subject);

    @Deprecated()
    public void setSubject(String subjectDN) throws IOException;

    public void setSubject(byte[] subjectDN) throws IOException;

    public void setSubjectKeyIdentifier(byte[] subjectKeyID);

    public void setAuthorityKeyIdentifier(byte[] authorityKeyID);

    public void setCertificateValid(Date certValid);

    public void setPrivateKeyValid(Date privateKeyValid);

    public void setSubjectPublicKeyAlgID(String oid) throws IOException;

    public void setSubjectPublicKey(PublicKey key);

    public void setSubjectPublicKey(byte[] key) throws IOException;

    public void setKeyUsage(boolean[] keyUsage);

    public void setExtendedKeyUsage(Set<String> keyPurposeSet) throws IOException;

    public void setMatchAllSubjectAltNames(boolean matchAllNames);

    public void setSubjectAlternativeNames(Collection<List<?>> names) throws IOException;

    public void addSubjectAlternativeName(int type, String name) throws IOException;

    public void addSubjectAlternativeName(int type, byte[] name) throws IOException;

    static boolean equalNames(Collection<?> object1, Collection<?> object2);

    static GeneralNameInterface makeGeneralNameInterface(int type, Object name) throws IOException;

    public void setNameConstraints(byte[] bytes) throws IOException;

    public void setBasicConstraints(int minMaxPathLen);

    public void setPolicy(Set<String> certPolicySet) throws IOException;

    public void setPathToNames(Collection<List<?>> names) throws IOException;

    void setPathToNamesInternal(Set<GeneralNameInterface> names);

    public void addPathToName(int type, String name) throws IOException;

    public void addPathToName(int type, byte[] name) throws IOException;

    public X509Certificate getCertificate();

    public BigInteger getSerialNumber();

    public X500Principal getIssuer();

    @Deprecated()
    public String getIssuerAsString();

    public byte[] getIssuerAsBytes() throws IOException;

    public X500Principal getSubject();

    @Deprecated()
    public String getSubjectAsString();

    public byte[] getSubjectAsBytes() throws IOException;

    public byte[] getSubjectKeyIdentifier();

    public byte[] getAuthorityKeyIdentifier();

    public Date getCertificateValid();

    public Date getPrivateKeyValid();

    public String getSubjectPublicKeyAlgID();

    public PublicKey getSubjectPublicKey();

    public boolean[] getKeyUsage();

    public Set<String> getExtendedKeyUsage();

    public boolean getMatchAllSubjectAltNames();

    public Collection<List<?>> getSubjectAlternativeNames();

    public byte[] getNameConstraints();

    public int getBasicConstraints();

    public Set<String> getPolicy();

    public Collection<List<?>> getPathToNames();

    public String toString();

    public boolean match(Certificate cert);

    public Object clone();
}
