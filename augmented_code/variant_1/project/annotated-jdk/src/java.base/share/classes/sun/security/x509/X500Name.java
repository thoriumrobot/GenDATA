/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.x509;

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
import java.lang.reflect.*;
    @Positive
import java.io.IOException;
    @Positive
import java.security.PrivilegedExceptionAction;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.Principal;
    @Positive
import java.util.*;
    @Positive
import java.util.StringJoiner;
    @Positive
import sun.security.util.*;
    @Positive
import javax.security.auth.x500.X500Principal;

    @Positive
public class X500Name implements GeneralNameInterface, Principal {

    @Positive
    public X500Name(String dname) throws IOException {
    @Positive
    }

    @Positive
    public X500Name(String dname, Map<String, String> keywordMap) throws IOException {
    @Positive
    }

    @Positive
    public X500Name(String dname, String format) throws IOException {
    @Positive
    }

    @Positive
    public X500Name(String commonName, String organizationUnit, String organizationName, String country) throws IOException {
    @Positive
    }

    @Positive
    public X500Name(String commonName, String organizationUnit, String organizationName, String localityName, String stateName, String country) throws IOException {
    @Positive
    }

    @Positive
    public X500Name(RDN[] rdnArray) throws IOException {
    @Positive
    }

    @Positive
    public X500Name(DerValue value) throws IOException {
    @Positive
    }

    @Positive
    public X500Name(DerInputStream in) throws IOException {
    @Positive
    }

    @Positive
    public X500Name(byte[] name) throws IOException {
    @Positive
    }

    @Positive
    public List<RDN> rdns();

    @Positive
    public int size();

    @Positive
    public List<AVA> allAvas();

    @Positive
    public int avaSize();

    @Positive
    public boolean isEmpty();

    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int getType();

    @Positive
    public String getCountry() throws IOException;

    @Positive
    public String getOrganization() throws IOException;

    @Positive
    public String getOrganizationalUnit() throws IOException;

    @Positive
    public String getCommonName() throws IOException;

    @Positive
    public String getLocality() throws IOException;

    @Positive
    public String getState() throws IOException;

    @Positive
    public String getDomain() throws IOException;

    @Positive
    public String getDNQualifier() throws IOException;

    @Positive
    public String getSurname() throws IOException;

    @Positive
    public String getGivenName() throws IOException;

    @Positive
    public String getInitials() throws IOException;

    @Positive
    public String getGeneration() throws IOException;

    @Positive
    public String getIP() throws IOException;

    @Positive
    public String toString();

    @Positive
    public String getRFC1779Name();

    @Positive
    public String getRFC1779Name(Map<String, String> oidMap) throws IllegalArgumentException;

    @Positive
    public String getRFC2253Name();

    @Positive
    public String getRFC2253Name(Map<String, String> oidMap);

    @Positive
    public String getRFC2253CanonicalName();

    @Positive
    public String getName();

    @Positive
    public DerValue findMostSpecificAttribute(ObjectIdentifier attribute);

    @Positive
    @Deprecated
    @Positive
    public void emit(DerOutputStream out) throws IOException;

    @Positive
    public void encode(DerOutputStream out) throws IOException;

    @Positive
    public byte[] getEncodedInternal() throws IOException;

    @Positive
    public byte[] getEncoded() throws IOException;

    @Positive
    static int countQuotes(String string, int from, int to);

    @Positive
    public static final ObjectIdentifier commonName_oid;

    @Positive
    public static final ObjectIdentifier SURNAME_OID;

    @Positive
    public static final ObjectIdentifier SERIALNUMBER_OID;

    @Positive
    public static final ObjectIdentifier countryName_oid;

    @Positive
    public static final ObjectIdentifier localityName_oid;

    @Positive
    public static final ObjectIdentifier stateName_oid;

    @Positive
    public static final ObjectIdentifier streetAddress_oid;

    @Positive
    public static final ObjectIdentifier orgName_oid;

    @Positive
    public static final ObjectIdentifier orgUnitName_oid;

    @Positive
    public static final ObjectIdentifier title_oid;

    @Positive
    public static final ObjectIdentifier GIVENNAME_OID;

    @Positive
    public static final ObjectIdentifier INITIALS_OID;

    @Positive
    public static final ObjectIdentifier GENERATIONQUALIFIER_OID;

    @Positive
    public static final ObjectIdentifier DNQUALIFIER_OID;

    @Positive
    public static final ObjectIdentifier ipAddress_oid;

    @Positive
    public static final ObjectIdentifier DOMAIN_COMPONENT_OID;

    @Positive
    public static final ObjectIdentifier userid_oid;

    @Positive
    public int constrains(GeneralNameInterface inputName) throws UnsupportedOperationException;

    @Positive
    public int subtreeDepth() throws UnsupportedOperationException;

    @Positive
    public X500Name commonAncestor(X500Name other);

    @Positive
    public X500Principal asX500Principal();

    @Positive
    public static X500Name asX500Name(X500Principal p);
    @Positive
}
