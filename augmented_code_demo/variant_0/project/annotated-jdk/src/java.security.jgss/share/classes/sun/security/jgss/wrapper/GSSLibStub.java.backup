/*
 * Copyright (c) 2005, 2019, Oracle and/or its affiliates. All rights reserved.
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
import java.util.Hashtable;
import org.ietf.jgss.Oid;
import org.ietf.jgss.GSSName;
import org.ietf.jgss.ChannelBinding;
import org.ietf.jgss.MessageProp;
import org.ietf.jgss.GSSException;
import sun.security.jgss.GSSUtil;

class GSSLibStub {

    static native boolean init(String lib, boolean debug);

    static native Oid[] indicateMechs();

    native Oid[] inquireNamesForMech() throws GSSException;

    native void releaseName(long pName);

    native long importName(byte[] name, Oid type);

    native boolean compareName(long pName1, long pName2);

    native long canonicalizeName(long pName);

    native byte[] exportName(long pName) throws GSSException;

    native Object[] displayName(long pName) throws GSSException;

    native long acquireCred(long pName, int lifetime, int usage) throws GSSException;

    native long releaseCred(long pCred);

    native long getCredName(long pCred);

    native int getCredTime(long pCred);

    native int getCredUsage(long pCred);

    native NativeGSSContext importContext(byte[] interProcToken);

    native byte[] initContext(long pCred, long targetName, ChannelBinding cb, byte[] inToken, NativeGSSContext context);

    native byte[] acceptContext(long pCred, ChannelBinding cb, byte[] inToken, NativeGSSContext context);

    native long[] inquireContext(long pContext);

    native Oid getContextMech(long pContext);

    native long getContextName(long pContext, boolean isSrc);

    native int getContextTime(long pContext);

    native long deleteContext(long pContext);

    native int wrapSizeLimit(long pContext, int flags, int qop, int outSize);

    native byte[] exportContext(long pContext);

    native byte[] getMic(long pContext, int qop, byte[] msg);

    native void verifyMic(long pContext, byte[] token, byte[] msg, MessageProp prop);

    native byte[] wrap(long pContext, byte[] msg, MessageProp prop);

    native byte[] unwrap(long pContext, byte[] msgToken, MessageProp prop);

    static GSSLibStub getInstance(Oid mech) throws GSSException;

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();

    Oid getMech();
}
