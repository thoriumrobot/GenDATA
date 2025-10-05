/*
    @Positive
 * Copyright (c) 1996, 2011, Oracle and/or its affiliates. All rights reserved.
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
package sun.rmi.transport;

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
import java.io.IOException;
    @Positive
import java.io.ObjectInput;
    @Positive
import java.io.ObjectOutput;
    @Positive
import java.rmi.Remote;
    @Positive
import java.rmi.RemoteException;
    @Positive
import java.rmi.server.ObjID;
    @Positive
import java.rmi.server.RMIClientSocketFactory;
    @Positive
import java.rmi.server.RMIServerSocketFactory;
    @Positive
import java.util.Arrays;
    @Positive
import sun.rmi.transport.tcp.TCPEndpoint;

    @Positive
public class LiveRef implements Cloneable {

    @Positive
    public LiveRef(ObjID objID, Endpoint endpoint, boolean isLocal) {
    @Positive
    }

    @Positive
    public LiveRef(int port) {
    @Positive
    }

    @Positive
    public LiveRef(int port, RMIClientSocketFactory csf, RMIServerSocketFactory ssf) {
    @Positive
    }

    @Positive
    public LiveRef(ObjID objID, int port) {
    @Positive
    }

    @Positive
    public LiveRef(ObjID objID, int port, RMIClientSocketFactory csf, RMIServerSocketFactory ssf) {
    @Positive
    }

    @Positive
    public Object clone();

    @Positive
    public int getPort();

    @Positive
    public RMIClientSocketFactory getClientSocketFactory();

    @Positive
    public RMIServerSocketFactory getServerSocketFactory();

    @Positive
    public void exportObject(Target target) throws RemoteException;

    @Positive
    public Channel getChannel() throws RemoteException;

    @Positive
    public ObjID getObjID();

    @Positive
    Endpoint getEndpoint();

    @Positive
    public String toString();

    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public boolean remoteEquals(Object obj);

    @Positive
    public void write(ObjectOutput out, boolean useNewFormat) throws IOException;

    @Positive
    public static LiveRef read(ObjectInput in, boolean useNewFormat) throws IOException, ClassNotFoundException;
    @Positive
}

// CFWR semantic augmentation - variant 0
