/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1998, 2020, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.jdi;

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
import java.lang.ref.Reference;
    @Positive
import java.lang.ref.ReferenceQueue;
    @Positive
import java.lang.ref.SoftReference;
    @Positive
import java.text.MessageFormat;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Collections;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.Set;
    @Positive
import java.util.function.Consumer;
    @Positive
import com.sun.jdi.BooleanType;
    @Positive
import com.sun.jdi.BooleanValue;
    @Positive
import com.sun.jdi.ByteType;
    @Positive
import com.sun.jdi.ByteValue;
    @Positive
import com.sun.jdi.CharType;
    @Positive
import com.sun.jdi.CharValue;
    @Positive
import com.sun.jdi.ClassLoaderReference;
    @Positive
import com.sun.jdi.ClassNotLoadedException;
    @Positive
import com.sun.jdi.DoubleType;
    @Positive
import com.sun.jdi.DoubleValue;
    @Positive
import com.sun.jdi.FloatType;
    @Positive
import com.sun.jdi.FloatValue;
    @Positive
import com.sun.jdi.IntegerType;
    @Positive
import com.sun.jdi.IntegerValue;
    @Positive
import com.sun.jdi.InternalException;
    @Positive
import com.sun.jdi.LongType;
    @Positive
import com.sun.jdi.LongValue;
    @Positive
import com.sun.jdi.ModuleReference;
    @Positive
import com.sun.jdi.ObjectCollectedException;
    @Positive
import com.sun.jdi.PathSearchingVirtualMachine;
    @Positive
import com.sun.jdi.PrimitiveType;
    @Positive
import com.sun.jdi.ReferenceType;
    @Positive
import com.sun.jdi.ShortType;
    @Positive
import com.sun.jdi.ShortValue;
    @Positive
import com.sun.jdi.StringReference;
    @Positive
import com.sun.jdi.ThreadGroupReference;
    @Positive
import com.sun.jdi.ThreadReference;
    @Positive
import com.sun.jdi.Type;
    @Positive
import com.sun.jdi.VMDisconnectedException;
    @Positive
import com.sun.jdi.VirtualMachine;
    @Positive
import com.sun.jdi.VirtualMachineManager;
    @Positive
import com.sun.jdi.VoidType;
    @Positive
import com.sun.jdi.VoidValue;
    @Positive
import com.sun.jdi.connect.spi.Connection;
    @Positive
import com.sun.jdi.event.EventQueue;
    @Positive
import com.sun.jdi.request.BreakpointRequest;
    @Positive
import com.sun.jdi.request.EventRequest;
    @Positive
import com.sun.jdi.request.EventRequestManager;

    @Positive
class VirtualMachineImpl extends MirrorImpl implements PathSearchingVirtualMachine, ThreadListener {

    @Positive
    public final int sizeofFieldRef;

    @Positive
    public final int sizeofMethodRef;

    @Positive
    public final int sizeofObjectRef;

    @Positive
    public final int sizeofClassRef;

    @Positive
    public final int sizeofFrameRef;

    @Positive
    public final int sizeofModuleRef;

    @Positive
    void waitInitCompletion();

    @Positive
    VMState state();

    @Positive
    public boolean threadResumable(ThreadAction action);

    @Positive
    EventRequestManagerImpl getInternalEventRequestManager();

    @Positive
    void validateVM();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    public List<ModuleReference> allModules();

    @Positive
    public List<ReferenceType> classesByName(String className);

    @Positive
    List<ReferenceType> classesBySignature(String signature);

    @Positive
    public List<ReferenceType> allClasses();

    @Positive
    public void forEachClass(Consumer<ReferenceType> action);

    @Positive
    public void redefineClasses(Map<? extends ReferenceType, byte[]> classToBytes);

    @Positive
    public List<ThreadReference> allThreads();

    @Positive
    public List<ThreadGroupReference> topLevelThreadGroups();

    @Positive
    PacketStream sendResumingCommand(CommandSender sender);

    @Positive
    void notifySuspend();

    @Positive
    public void suspend();

    @Positive
    public void resume();

    @Positive
    public EventQueue eventQueue();

    @Positive
    public EventRequestManager eventRequestManager();

    @Positive
    EventRequestManagerImpl eventRequestManagerImpl();

    @Positive
    public BooleanValue mirrorOf(boolean value);

    @Positive
    public ByteValue mirrorOf(byte value);

    @Positive
    public CharValue mirrorOf(char value);

    @Positive
    public ShortValue mirrorOf(short value);

    @Positive
    public IntegerValue mirrorOf(int value);

    @Positive
    public LongValue mirrorOf(long value);

    @Positive
    public FloatValue mirrorOf(float value);

    @Positive
    public DoubleValue mirrorOf(double value);

    @Positive
    public StringReference mirrorOf(String value);

    @Positive
    public VoidValue mirrorOfVoid();

    @Positive
    public long[] instanceCounts(List<? extends ReferenceType> classes);

    @Positive
    public void dispose();

    @Positive
    public void exit(int exitCode);

    @Positive
    public Process process();

    @Positive
    public String description();

    @Positive
    public String version();

    @Positive
    public String name();

    @Positive
    public boolean canWatchFieldModification();

    @Positive
    public boolean canWatchFieldAccess();

    @Positive
    public boolean canGetBytecodes();

    @Positive
    public boolean canGetSyntheticAttribute();

    @Positive
    public boolean canGetOwnedMonitorInfo();

    @Positive
    public boolean canGetCurrentContendedMonitor();

    @Positive
    public boolean canGetMonitorInfo();

    @Positive
    boolean canGet1_5LanguageFeatures();

    @Positive
    public boolean canUseInstanceFilters();

    @Positive
    public boolean canRedefineClasses();

    @Positive
    @Deprecated()
    @Positive
    public boolean canAddMethod();

    @Positive
    @Deprecated()
    @Positive
    public boolean canUnrestrictedlyRedefineClasses();

    @Positive
    public boolean canPopFrames();

    @Positive
    public boolean canGetMethodReturnValues();

    @Positive
    public boolean canGetInstanceInfo();

    @Positive
    public boolean canUseSourceNameFilters();

    @Positive
    public boolean canForceEarlyReturn();

    @Positive
    public boolean canBeModified();

    @Positive
    public boolean canGetSourceDebugExtension();

    @Positive
    public boolean canGetClassFileVersion();

    @Positive
    public boolean canGetConstantPool();

    @Positive
    public boolean canRequestVMDeathEvent();

    @Positive
    public boolean canRequestMonitorEvents();

    @Positive
    public boolean canGetMonitorFrameInfo();

    @Positive
    public boolean canGetModuleInfo();

    @Positive
    public void setDebugTraceMode(int traceFlags);

    @Positive
    void printTrace(String string);

    @Positive
    void printReceiveTrace(int depth, String string);

    @Positive
    synchronized void removeReferenceType(String signature);

    @Positive
    ReferenceTypeImpl referenceType(long ref, byte tag);

    @Positive
    ClassTypeImpl classType(long ref);

    @Positive
    InterfaceTypeImpl interfaceType(long ref);

    @Positive
    ArrayTypeImpl arrayType(long ref);

    @Positive
    ReferenceTypeImpl referenceType(long id, int tag, String signature);

    @Positive
    ModuleReference getModule(long id);

    @Positive
    void sendToTarget(Packet packet);

    @Positive
    void waitForTargetReply(Packet packet);

    @Positive
    Type findBootType(String signature) throws ClassNotLoadedException;

    @Positive
    BooleanType theBooleanType();

    @Positive
    ByteType theByteType();

    @Positive
    CharType theCharType();

    @Positive
    ShortType theShortType();

    @Positive
    IntegerType theIntegerType();

    @Positive
    LongType theLongType();

    @Positive
    FloatType theFloatType();

    @Positive
    DoubleType theDoubleType();

    @Positive
    VoidType theVoidType();

    @Positive
    PrimitiveType primitiveTypeMirror(byte tag);

    @Positive
    synchronized ObjectReferenceImpl objectMirror(long id, int tag);

    @Positive
    synchronized void removeObjectMirror(ObjectReferenceImpl object);

    @Positive
    synchronized void removeObjectMirror(SoftObjectReference ref);

    @Positive
    ObjectReferenceImpl objectMirror(long id);

    @Positive
    StringReferenceImpl stringMirror(long id);

    @Positive
    ArrayReferenceImpl arrayMirror(long id);

    @Positive
    ThreadReferenceImpl threadMirror(long id);

    @Positive
    ThreadGroupReferenceImpl threadGroupMirror(long id);

    @Positive
    ClassLoaderReferenceImpl classLoaderMirror(long id);

    @Positive
    ClassObjectReferenceImpl classObjectMirror(long id);

    @Positive
    ModuleReferenceImpl moduleMirror(long id);

    @Positive
    public List<String> classPath();

    @Positive
    public List<String> bootClassPath();

    @Positive
    public String baseDirectory();

    @Positive
    public void setDefaultStratum(String stratum);

    @Positive
    public String getDefaultStratum();

    @Positive
    ThreadGroup threadGroupForJDI();

    @Positive
    static private class SoftObjectReference extends SoftReference<ObjectReferenceImpl> {

    @Positive
        int count();

    @Positive
        void incrementCount();

    @Positive
        Long key();

    @Positive
        ObjectReferenceImpl object();
    @Positive
    }
    @Positive
}
