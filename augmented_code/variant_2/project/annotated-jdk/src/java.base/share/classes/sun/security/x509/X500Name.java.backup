/*
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.x509;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.lang.reflect.*;
import java.io.IOException;
import java.security.PrivilegedExceptionAction;
import java.security.AccessController;
import java.security.Principal;
import java.util.*;
import java.util.StringJoiner;
import sun.security.util.*;
import javax.security.auth.x500.X500Principal;

public class X500Name implements GeneralNameInterface, Principal {

    public X500Name(String dname) throws IOException {
    }

    public X500Name(String dname, Map<String, String> keywordMap) throws IOException {
    }

    public X500Name(String dname, String format) throws IOException {
    }

    public X500Name(String commonName, String organizationUnit, String organizationName, String country) throws IOException {
    }

    public X500Name(String commonName, String organizationUnit, String organizationName, String localityName, String stateName, String country) throws IOException {
    }

    public X500Name(RDN[] rdnArray) throws IOException {
    }

    public X500Name(DerValue value) throws IOException {
    }

    public X500Name(DerInputStream in) throws IOException {
    }

    public X500Name(byte[] name) throws IOException {
    }

    public List<RDN> rdns();

    public int size();

    public List<AVA> allAvas();

    public int avaSize();

    public boolean isEmpty();

    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int getType();

    public String getCountry() throws IOException;

    public String getOrganization() throws IOException;

    public String getOrganizationalUnit() throws IOException;

    public String getCommonName() throws IOException;

    public String getLocality() throws IOException;

    public String getState() throws IOException;

    public String getDomain() throws IOException;

    public String getDNQualifier() throws IOException;

    public String getSurname() throws IOException;

    public String getGivenName() throws IOException;

    public String getInitials() throws IOException;

    public String getGeneration() throws IOException;

    public String getIP() throws IOException;

    public String toString();

    public String getRFC1779Name();

    public String getRFC1779Name(Map<String, String> oidMap) throws IllegalArgumentException;

    public String getRFC2253Name();

    public String getRFC2253Name(Map<String, String> oidMap);

    public String getRFC2253CanonicalName();

    public String getName();

    public DerValue findMostSpecificAttribute(ObjectIdentifier attribute);

    @Deprecated
    public void emit(DerOutputStream out) throws IOException;

    public void encode(DerOutputStream out) throws IOException;

    public byte[] getEncodedInternal() throws IOException;

    public byte[] getEncoded() throws IOException;

    static int countQuotes(String string, int from, int to);

    public static final ObjectIdentifier commonName_oid;

    public static final ObjectIdentifier SURNAME_OID;

    public static final ObjectIdentifier SERIALNUMBER_OID;

    public static final ObjectIdentifier countryName_oid;

    public static final ObjectIdentifier localityName_oid;

    public static final ObjectIdentifier stateName_oid;

    public static final ObjectIdentifier streetAddress_oid;

    public static final ObjectIdentifier orgName_oid;

    public static final ObjectIdentifier orgUnitName_oid;

    public static final ObjectIdentifier title_oid;

    public static final ObjectIdentifier GIVENNAME_OID;

    public static final ObjectIdentifier INITIALS_OID;

    public static final ObjectIdentifier GENERATIONQUALIFIER_OID;

    public static final ObjectIdentifier DNQUALIFIER_OID;

    public static final ObjectIdentifier ipAddress_oid;

    public static final ObjectIdentifier DOMAIN_COMPONENT_OID;

    public static final ObjectIdentifier userid_oid;

    public int constrains(GeneralNameInterface inputName) throws UnsupportedOperationException;

    public int subtreeDepth() throws UnsupportedOperationException;

    public X500Name commonAncestor(X500Name other);

    public X500Principal asX500Principal();

    public static X500Name asX500Name(X500Principal p);
}
