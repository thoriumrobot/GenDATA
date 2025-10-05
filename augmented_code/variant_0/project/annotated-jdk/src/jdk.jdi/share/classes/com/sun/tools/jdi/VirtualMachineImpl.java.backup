/*
 * Copyright (c) 1998, 2020, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.jdi;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.lang.ref.SoftReference;
import java.text.MessageFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;
import com.sun.jdi.BooleanType;
import com.sun.jdi.BooleanValue;
import com.sun.jdi.ByteType;
import com.sun.jdi.ByteValue;
import com.sun.jdi.CharType;
import com.sun.jdi.CharValue;
import com.sun.jdi.ClassLoaderReference;
import com.sun.jdi.ClassNotLoadedException;
import com.sun.jdi.DoubleType;
import com.sun.jdi.DoubleValue;
import com.sun.jdi.FloatType;
import com.sun.jdi.FloatValue;
import com.sun.jdi.IntegerType;
import com.sun.jdi.IntegerValue;
import com.sun.jdi.InternalException;
import com.sun.jdi.LongType;
import com.sun.jdi.LongValue;
import com.sun.jdi.ModuleReference;
import com.sun.jdi.ObjectCollectedException;
import com.sun.jdi.PathSearchingVirtualMachine;
import com.sun.jdi.PrimitiveType;
import com.sun.jdi.ReferenceType;
import com.sun.jdi.ShortType;
import com.sun.jdi.ShortValue;
import com.sun.jdi.StringReference;
import com.sun.jdi.ThreadGroupReference;
import com.sun.jdi.ThreadReference;
import com.sun.jdi.Type;
import com.sun.jdi.VMDisconnectedException;
import com.sun.jdi.VirtualMachine;
import com.sun.jdi.VirtualMachineManager;
import com.sun.jdi.VoidType;
import com.sun.jdi.VoidValue;
import com.sun.jdi.connect.spi.Connection;
import com.sun.jdi.event.EventQueue;
import com.sun.jdi.request.BreakpointRequest;
import com.sun.jdi.request.EventRequest;
import com.sun.jdi.request.EventRequestManager;

class VirtualMachineImpl extends MirrorImpl implements PathSearchingVirtualMachine, ThreadListener {

    public final int sizeofFieldRef;

    public final int sizeofMethodRef;

    public final int sizeofObjectRef;

    public final int sizeofClassRef;

    public final int sizeofFrameRef;

    public final int sizeofModuleRef;

    void waitInitCompletion();

    VMState state();

    public boolean threadResumable(ThreadAction action);

    EventRequestManagerImpl getInternalEventRequestManager();

    void validateVM();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();

    public List<ModuleReference> allModules();

    public List<ReferenceType> classesByName(String className);

    List<ReferenceType> classesBySignature(String signature);

    public List<ReferenceType> allClasses();

    public void forEachClass(Consumer<ReferenceType> action);

    public void redefineClasses(Map<? extends ReferenceType, byte[]> classToBytes);

    public List<ThreadReference> allThreads();

    public List<ThreadGroupReference> topLevelThreadGroups();

    PacketStream sendResumingCommand(CommandSender sender);

    void notifySuspend();

    public void suspend();

    public void resume();

    public EventQueue eventQueue();

    public EventRequestManager eventRequestManager();

    EventRequestManagerImpl eventRequestManagerImpl();

    public BooleanValue mirrorOf(boolean value);

    public ByteValue mirrorOf(byte value);

    public CharValue mirrorOf(char value);

    public ShortValue mirrorOf(short value);

    public IntegerValue mirrorOf(int value);

    public LongValue mirrorOf(long value);

    public FloatValue mirrorOf(float value);

    public DoubleValue mirrorOf(double value);

    public StringReference mirrorOf(String value);

    public VoidValue mirrorOfVoid();

    public long[] instanceCounts(List<? extends ReferenceType> classes);

    public void dispose();

    public void exit(int exitCode);

    public Process process();

    public String description();

    public String version();

    public String name();

    public boolean canWatchFieldModification();

    public boolean canWatchFieldAccess();

    public boolean canGetBytecodes();

    public boolean canGetSyntheticAttribute();

    public boolean canGetOwnedMonitorInfo();

    public boolean canGetCurrentContendedMonitor();

    public boolean canGetMonitorInfo();

    boolean canGet1_5LanguageFeatures();

    public boolean canUseInstanceFilters();

    public boolean canRedefineClasses();

    @Deprecated()
    public boolean canAddMethod();

    @Deprecated()
    public boolean canUnrestrictedlyRedefineClasses();

    public boolean canPopFrames();

    public boolean canGetMethodReturnValues();

    public boolean canGetInstanceInfo();

    public boolean canUseSourceNameFilters();

    public boolean canForceEarlyReturn();

    public boolean canBeModified();

    public boolean canGetSourceDebugExtension();

    public boolean canGetClassFileVersion();

    public boolean canGetConstantPool();

    public boolean canRequestVMDeathEvent();

    public boolean canRequestMonitorEvents();

    public boolean canGetMonitorFrameInfo();

    public boolean canGetModuleInfo();

    public void setDebugTraceMode(int traceFlags);

    void printTrace(String string);

    void printReceiveTrace(int depth, String string);

    synchronized void removeReferenceType(String signature);

    ReferenceTypeImpl referenceType(long ref, byte tag);

    ClassTypeImpl classType(long ref);

    InterfaceTypeImpl interfaceType(long ref);

    ArrayTypeImpl arrayType(long ref);

    ReferenceTypeImpl referenceType(long id, int tag, String signature);

    ModuleReference getModule(long id);

    void sendToTarget(Packet packet);

    void waitForTargetReply(Packet packet);

    Type findBootType(String signature) throws ClassNotLoadedException;

    BooleanType theBooleanType();

    ByteType theByteType();

    CharType theCharType();

    ShortType theShortType();

    IntegerType theIntegerType();

    LongType theLongType();

    FloatType theFloatType();

    DoubleType theDoubleType();

    VoidType theVoidType();

    PrimitiveType primitiveTypeMirror(byte tag);

    synchronized ObjectReferenceImpl objectMirror(long id, int tag);

    synchronized void removeObjectMirror(ObjectReferenceImpl object);

    synchronized void removeObjectMirror(SoftObjectReference ref);

    ObjectReferenceImpl objectMirror(long id);

    StringReferenceImpl stringMirror(long id);

    ArrayReferenceImpl arrayMirror(long id);

    ThreadReferenceImpl threadMirror(long id);

    ThreadGroupReferenceImpl threadGroupMirror(long id);

    ClassLoaderReferenceImpl classLoaderMirror(long id);

    ClassObjectReferenceImpl classObjectMirror(long id);

    ModuleReferenceImpl moduleMirror(long id);

    public List<String> classPath();

    public List<String> bootClassPath();

    public String baseDirectory();

    public void setDefaultStratum(String stratum);

    public String getDefaultStratum();

    ThreadGroup threadGroupForJDI();

    static private class SoftObjectReference extends SoftReference<ObjectReferenceImpl> {

        int count();

        void incrementCount();

        Long key();

        ObjectReferenceImpl object();
    }
}
