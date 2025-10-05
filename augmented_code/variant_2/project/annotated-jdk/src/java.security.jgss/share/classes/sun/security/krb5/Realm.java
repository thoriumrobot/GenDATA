/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2000, 2019, Oracle and/or its affiliates. All rights reserved.
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
import sun.security.action.GetBooleanAction;
    @Positive
import sun.security.krb5.internal.Krb5;
    @Positive
import sun.security.util.*;
    @Positive
import java.io.IOException;
    @Positive
import java.util.*;
    @Positive
import sun.security.krb5.internal.util.KerberosString;

    @Positive
public class Realm implements Cloneable {

    @Positive
    public static final boolean AUTODEDUCEREALM;

    @Positive
    public Realm(String name) throws RealmException {
    @Positive
    }

    @Positive
    public static Realm getDefault() throws RealmException;

    @Positive
    public Object clone();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    public Realm(DerValue encoding) throws Asn1Exception, RealmException, IOException {
    @Positive
    }

    @Positive
    public String toString();

    @Positive
    public static String parseRealmAtSeparator(String name) throws RealmException;

    @Positive
    public static String parseRealmComponent(String name);

    @Positive
    protected static String parseRealm(String name) throws RealmException;

    @Positive
    protected static boolean isValidRealmString(String name);

    @Positive
    public byte[] asn1Encode() throws Asn1Exception, IOException;

    @Positive
    public static Realm parse(DerInputStream data, byte explicitTag, boolean optional) throws Asn1Exception, IOException, RealmException;

    @Positive
    public static String[] getRealmsList(String cRealm, String sRealm);
    @Positive
}
