/*
 * Copyright (c) 2005, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.jgss.wrapper;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.ietf.jgss.*;
import java.security.Provider;
import java.security.Security;
import java.io.IOException;
import sun.security.krb5.Realm;
import sun.security.jgss.GSSUtil;
import sun.security.util.ObjectIdentifier;
import sun.security.util.DerInputStream;
import sun.security.util.DerOutputStream;
import sun.security.jgss.GSSUtil;
import sun.security.jgss.GSSExceptionImpl;
import sun.security.jgss.spi.GSSNameSpi;
import javax.security.auth.kerberos.ServicePermission;

public class GSSNameElement implements GSSNameSpi {

    public String getKrbName() throws GSSException;

    public Provider getProvider();

    public boolean equals(GSSNameSpi other) throws GSSException;

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object other);

    public int hashCode();

    public byte[] export() throws GSSException;

    public Oid getMechanism();

    public String toString();

    public Oid getStringNameType();

    public boolean isAnonymousName();

    public void dispose();

    @SuppressWarnings("deprecation")
    protected void finalize() throws Throwable;
}
