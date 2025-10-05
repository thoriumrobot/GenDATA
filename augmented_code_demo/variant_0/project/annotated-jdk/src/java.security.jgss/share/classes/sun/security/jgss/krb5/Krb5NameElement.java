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
package sun.security.jgss.krb5;

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
import org.ietf.jgss.*;
    @Positive
import sun.security.jgss.spi.*;
    @Positive
import sun.security.krb5.PrincipalName;
    @Positive
import sun.security.krb5.Realm;
    @Positive
import sun.security.krb5.KrbException;
    @Positive
import javax.security.auth.kerberos.ServicePermission;
    @Positive
import java.net.InetAddress;
    @Positive
import java.net.UnknownHostException;
    @Positive
import java.security.Provider;
    @Positive
import java.util.Locale;
    @Positive
import static java.nio.charset.StandardCharsets.UTF_8;

    @Positive
public class Krb5NameElement implements GSSNameSpi {

    @Positive
    static Krb5NameElement getInstance(String gssNameStr, Oid gssNameType) throws GSSException;

    @Positive
    public static Krb5NameElement getInstance(PrincipalName principalName);

    @Positive
    public final PrincipalName getKrb5PrincipalName();

    @Positive
    public boolean equals(GSSNameSpi other) throws GSSException;

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object another);

    @Positive
    public int hashCode();

    @Positive
    public byte[] export() throws GSSException;

    @Positive
    public Oid getMechanism();

    @Positive
    public String toString();

    @Positive
    public Oid getGSSNameType();

    @Positive
    public Oid getStringNameType();

    @Positive
    public boolean isAnonymousName();

    @Positive
    public Provider getProvider();
    @Positive
}

// CFWR semantic augmentation - variant 0
