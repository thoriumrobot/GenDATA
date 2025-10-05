/*
    @Positive
 * Copyright (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.rmi.registry;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.rmi.RemoteException;
    @Positive
import java.rmi.server.ObjID;
    @Positive
import java.rmi.server.RMIClientSocketFactory;
    @Positive
import java.rmi.server.RMIServerSocketFactory;
    @Positive
import java.rmi.server.RemoteRef;
    @Positive
import java.rmi.server.UnicastRemoteObject;
    @Positive
import sun.rmi.registry.RegistryImpl;
    @Positive
import sun.rmi.server.UnicastRef2;
    @Positive
import sun.rmi.server.UnicastRef;
    @Positive
import sun.rmi.server.Util;
    @Positive
import sun.rmi.transport.LiveRef;
    @Positive
import sun.rmi.transport.tcp.TCPEndpoint;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public final class LocateRegistry {

    @Positive
    public static Registry getRegistry() throws RemoteException;

    @Positive
    public static Registry getRegistry(int port) throws RemoteException;

    @Positive
    public static Registry getRegistry(String host) throws RemoteException;

    @Positive
    public static Registry getRegistry(String host, int port) throws RemoteException;

    @Positive
    public static Registry getRegistry(String host, int port, RMIClientSocketFactory csf) throws RemoteException;

    @Positive
    public static Registry createRegistry(int port) throws RemoteException;

    @Positive
    public static Registry createRegistry(int port, RMIClientSocketFactory csf, RMIServerSocketFactory ssf) throws RemoteException;
    @Positive
}

// CFWR semantic augmentation - variant 1
