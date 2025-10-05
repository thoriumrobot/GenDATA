/*
    @Positive
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.krb5;

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
import sun.security.krb5.internal.*;
    @Positive
import sun.security.util.*;
    @Positive
import java.net.*;
    @Positive
import java.util.Vector;
    @Positive
import java.util.Locale;
    @Positive
import java.io.IOException;
    @Positive
import java.math.BigInteger;
    @Positive
import java.util.Arrays;
    @Positive
import sun.security.krb5.internal.ccache.CCacheOutputStream;
    @Positive
import sun.security.krb5.internal.util.KerberosString;

    @Positive
public class PrincipalName implements Cloneable {

    @Positive
    public static final int KRB_NT_UNKNOWN;

    @Positive
    public static final int KRB_NT_PRINCIPAL;

    @Positive
    public static final int KRB_NT_SRV_INST;

    @Positive
    public static final int KRB_NT_SRV_HST;

    @Positive
    public static final int KRB_NT_SRV_XHST;

    @Positive
    public static final int KRB_NT_UID;

    @Positive
    public static final int KRB_NT_ENTERPRISE;

    @Positive
    public static final String TGS_DEFAULT_SRV_NAME;

    @Positive
    public static final int TGS_DEFAULT_NT;

    @Positive
    public static final char NAME_COMPONENT_SEPARATOR;

    @Positive
    public static final char NAME_REALM_SEPARATOR;

    @Positive
    public static final char REALM_COMPONENT_SEPARATOR;

    @Positive
    public static final String NAME_COMPONENT_SEPARATOR_STR;

    @Positive
    public static final String NAME_REALM_SEPARATOR_STR;

    @Positive
    public static final String REALM_COMPONENT_SEPARATOR_STR;

    @Positive
    public PrincipalName(int nameType, String[] nameStrings, Realm nameRealm) {
    @Positive
    }

    @Positive
    public PrincipalName(String[] nameParts, String realm) throws RealmException {
    @Positive
    }

    @Positive
    public Object clone();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    public PrincipalName(DerValue encoding, Realm realm) throws Asn1Exception, IOException {
    @Positive
    }

    @Positive
    public static PrincipalName parse(DerInputStream data, byte explicitTag, boolean optional, Realm realm) throws Asn1Exception, IOException, RealmException;

    @Positive
    public PrincipalName(String name, int type, String realm) throws RealmException {
    @Positive
    }

    @Positive
    public PrincipalName(String name, int type) throws RealmException {
    @Positive
    }

    @Positive
    public PrincipalName(String name) throws RealmException {
    @Positive
    }

    @Positive
    public PrincipalName(String name, String realm) throws RealmException {
    @Positive
    }

    @Positive
    public static PrincipalName tgsService(String r1, String r2) throws KrbException;

    @Positive
    public String getRealmAsString();

    @Positive
    public String getPrincipalNameAsString();

    @Positive
    public int hashCode();

    @Positive
    public String getName();

    @Positive
    public int getNameType();

    @Positive
    public String[] getNameStrings();

    @Positive
    public byte[][] toByteArray();

    @Positive
    public String getRealmString();

    @Positive
    public Realm getRealm();

    @Positive
    public String getSalt();

    @Positive
    public String toString();

    @Positive
    public String getNameString();

    @Positive
    public byte[] asn1Encode() throws Asn1Exception, IOException;

    @Positive
    public boolean match(PrincipalName pname);

    @Positive
    public void writePrincipal(CCacheOutputStream cos) throws IOException;

    @Positive
    public String getInstanceComponent();

    @Positive
    static String mapHostToRealm(String name);

    @Positive
    public boolean isRealmDeduced();
    @Positive
}

// CFWR semantic augmentation - variant 0
