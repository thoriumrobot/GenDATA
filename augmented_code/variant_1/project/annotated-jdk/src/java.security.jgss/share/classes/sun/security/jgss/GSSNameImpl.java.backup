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
package sun.security.jgss;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.ietf.jgss.*;
import sun.security.jgss.spi.*;
import java.util.Set;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Arrays;
import java.io.IOException;
import sun.security.util.ObjectIdentifier;
import sun.security.util.DerInputStream;
import sun.security.util.DerOutputStream;
import static java.nio.charset.StandardCharsets.UTF_8;

public class GSSNameImpl implements GSSName {

    static GSSNameImpl wrapElement(GSSManagerImpl gssManager, GSSNameSpi mechElement) throws GSSException;

    public GSSName canonicalize(Oid mech) throws GSSException;

    public boolean equals(GSSName other) throws GSSException;

    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object another);

    public byte[] export() throws GSSException;

    public String toString();

    public Oid getStringNameType() throws GSSException;

    public boolean isAnonymous();

    public boolean isMN();

    public synchronized GSSNameSpi getElement(Oid mechOid) throws GSSException;

    Set<GSSNameSpi> getElements();
}
