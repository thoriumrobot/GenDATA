/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
public class GSSLibStub {
/*
    @Positive
 * Copyright (c) 2005, 2019, Oracle and/or its affiliates. All rights reserved.
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
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
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
package sun.security.jgss.wrapper;

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
import java.util.Hashtable;
    @Positive
import org.ietf.jgss.Oid;
    @Positive
import org.ietf.jgss.GSSName;
    @Positive
import org.ietf.jgss.ChannelBinding;
    @Positive
import org.ietf.jgss.MessageProp;
    @Positive
import org.ietf.jgss.GSSException;
    @Positive
import sun.security.jgss.GSSUtil;

    @Positive
class GSSLibStub {

    @Positive
    static native boolean init(String lib, boolean debug);

    @Positive
    static native Oid[] indicateMechs();

    @Positive
    native Oid[] inquireNamesForMech() throws GSSException;

    @Positive
    native void releaseName(long pName);

    @Positive
    native long importName(byte[] name, Oid type);

    @Positive
    native boolean compareName(long pName1, long pName2);

    @Positive
    native long canonicalizeName(long pName);

    @Positive
    native byte[] exportName(long pName) throws GSSException;

    @Positive
    native Object[] displayName(long pName) throws GSSException;

    @Positive
    native long acquireCred(long pName, int lifetime, int usage) throws GSSException;

    @Positive
    native long releaseCred(long pCred);

    @Positive
    native long getCredName(long pCred);

    @Positive
    native int getCredTime(long pCred);

    @Positive
    native int getCredUsage(long pCred);

    @Positive
    native NativeGSSContext importContext(byte[] interProcToken);

    @Positive
    native byte[] initContext(long pCred, long targetName, ChannelBinding cb, byte[] inToken, NativeGSSContext context);

    @Positive
    native byte[] acceptContext(long pCred, ChannelBinding cb, byte[] inToken, NativeGSSContext context);

    @Positive
    native long[] inquireContext(long pContext);

    @Positive
    native Oid getContextMech(long pContext);

    @Positive
    native long getContextName(long pContext, boolean isSrc);

    @Positive
    native int getContextTime(long pContext);

    @Positive
    native long deleteContext(long pContext);

    @Positive
    native int wrapSizeLimit(long pContext, int flags, int qop, int outSize);

    @Positive
    native byte[] exportContext(long pContext);

    @Positive
    native byte[] getMic(long pContext, int qop, byte[] msg);

    @Positive
    native void verifyMic(long pContext, byte[] token, byte[] msg, MessageProp prop);

    @Positive
    native byte[] wrap(long pContext, byte[] msg, MessageProp prop);

    @Positive
    native byte[] unwrap(long pContext, byte[] msgToken, MessageProp prop);

    @Positive
    static GSSLibStub getInstance(Oid mech) throws GSSException;

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    Oid getMech();
    @Positive
}

}