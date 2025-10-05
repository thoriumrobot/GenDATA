/*
 * Copyright (c) 2000, 2013, Oracle and/or its affiliates. All rights reserved.
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
import java.util.*;
import sun.security.jgss.spnego.SpNegoCredElement;

public class GSSCredentialImpl implements GSSCredential {

    public GSSCredentialImpl() {
    }

    protected GSSCredentialImpl(GSSCredentialImpl src) {
    }

    public GSSCredentialImpl(GSSManagerImpl gssManager, GSSCredentialSpi mechElement) throws GSSException {
    }

    void init(GSSManagerImpl gssManager);

    public void dispose() throws GSSException;

    public GSSCredential impersonate(GSSName name) throws GSSException;

    public GSSName getName() throws GSSException;

    public GSSName getName(Oid mech) throws GSSException;

    public int getRemainingLifetime() throws GSSException;

    public int getRemainingInitLifetime(Oid mech) throws GSSException;

    public int getRemainingAcceptLifetime(Oid mech) throws GSSException;

    public int getUsage() throws GSSException;

    public int getUsage(Oid mech) throws GSSException;

    public Oid[] getMechs() throws GSSException;

    public void add(GSSName name, int initLifetime, int acceptLifetime, Oid mech, int usage) throws GSSException;

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object another);

    public int hashCode();

    public GSSCredentialSpi getElement(Oid mechOid, boolean initiate) throws GSSException;

    Set<GSSCredentialSpi> getElements();

    public String toString();

    static class SearchKey {

        public SearchKey(Oid mechOid, int usage) {
        }

        public Oid getMech();

        public int getUsage();

        public boolean equals(Object other);

        public int hashCode();
    }
}
