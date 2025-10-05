/*
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.krb5;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import sun.security.krb5.internal.*;
import sun.security.util.*;
import java.net.*;
import java.util.Vector;
import java.util.Locale;
import java.io.IOException;
import java.math.BigInteger;
import java.util.Arrays;
import sun.security.krb5.internal.ccache.CCacheOutputStream;
import sun.security.krb5.internal.util.KerberosString;

public class PrincipalName implements Cloneable {

    public static final int KRB_NT_UNKNOWN;

    public static final int KRB_NT_PRINCIPAL;

    public static final int KRB_NT_SRV_INST;

    public static final int KRB_NT_SRV_HST;

    public static final int KRB_NT_SRV_XHST;

    public static final int KRB_NT_UID;

    public static final int KRB_NT_ENTERPRISE;

    public static final String TGS_DEFAULT_SRV_NAME;

    public static final int TGS_DEFAULT_NT;

    public static final char NAME_COMPONENT_SEPARATOR;

    public static final char NAME_REALM_SEPARATOR;

    public static final char REALM_COMPONENT_SEPARATOR;

    public static final String NAME_COMPONENT_SEPARATOR_STR;

    public static final String NAME_REALM_SEPARATOR_STR;

    public static final String REALM_COMPONENT_SEPARATOR_STR;

    public PrincipalName(int nameType, String[] nameStrings, Realm nameRealm) {
    }

    public PrincipalName(String[] nameParts, String realm) throws RealmException {
    }

    public Object clone();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    public PrincipalName(DerValue encoding, Realm realm) throws Asn1Exception, IOException {
    }

    public static PrincipalName parse(DerInputStream data, byte explicitTag, boolean optional, Realm realm) throws Asn1Exception, IOException, RealmException;

    public PrincipalName(String name, int type, String realm) throws RealmException {
    }

    public PrincipalName(String name, int type) throws RealmException {
    }

    public PrincipalName(String name) throws RealmException {
    }

    public PrincipalName(String name, String realm) throws RealmException {
    }

    public static PrincipalName tgsService(String r1, String r2) throws KrbException;

    public String getRealmAsString();

    public String getPrincipalNameAsString();

    public int hashCode();

    public String getName();

    public int getNameType();

    public String[] getNameStrings();

    public byte[][] toByteArray();

    public String getRealmString();

    public Realm getRealm();

    public String getSalt();

    public String toString();

    public String getNameString();

    public byte[] asn1Encode() throws Asn1Exception, IOException;

    public boolean match(PrincipalName pname);

    public void writePrincipal(CCacheOutputStream cos) throws IOException;

    public String getInstanceComponent();

    static String mapHostToRealm(String name);

    public boolean isRealmDeduced();
}
