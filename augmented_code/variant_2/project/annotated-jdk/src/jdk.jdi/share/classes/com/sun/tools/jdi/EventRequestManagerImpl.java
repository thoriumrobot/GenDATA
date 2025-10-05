/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1998, 2017, Oracle and/or its affiliates. All rights reserved.
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
import java.util.ArrayList;
    @Positive
import java.util.Collections;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import com.sun.jdi.Field;
    @Positive
import com.sun.jdi.Location;
    @Positive
import com.sun.jdi.NativeMethodException;
    @Positive
import com.sun.jdi.ObjectReference;
    @Positive
import com.sun.jdi.ReferenceType;
    @Positive
import com.sun.jdi.ThreadReference;
    @Positive
import com.sun.jdi.VirtualMachine;
    @Positive
import com.sun.jdi.request.AccessWatchpointRequest;
    @Positive
import com.sun.jdi.request.BreakpointRequest;
    @Positive
import com.sun.jdi.request.ClassPrepareRequest;
    @Positive
import com.sun.jdi.request.ClassUnloadRequest;
    @Positive
import com.sun.jdi.request.DuplicateRequestException;
    @Positive
import com.sun.jdi.request.EventRequest;
    @Positive
import com.sun.jdi.request.EventRequestManager;
    @Positive
import com.sun.jdi.request.ExceptionRequest;
    @Positive
import com.sun.jdi.request.InvalidRequestStateException;
    @Positive
import com.sun.jdi.request.MethodEntryRequest;
    @Positive
import com.sun.jdi.request.MethodExitRequest;
    @Positive
import com.sun.jdi.request.ModificationWatchpointRequest;
    @Positive
import com.sun.jdi.request.MonitorContendedEnterRequest;
    @Positive
import com.sun.jdi.request.MonitorContendedEnteredRequest;
    @Positive
import com.sun.jdi.request.MonitorWaitRequest;
    @Positive
import com.sun.jdi.request.MonitorWaitedRequest;
    @Positive
import com.sun.jdi.request.StepRequest;
    @Positive
import com.sun.jdi.request.ThreadDeathRequest;
    @Positive
import com.sun.jdi.request.ThreadStartRequest;
    @Positive
import com.sun.jdi.request.VMDeathRequest;
    @Positive
import com.sun.jdi.request.WatchpointRequest;

    @Positive
@SuppressWarnings({ "unchecked", "rawtypes" })
    @Positive
class EventRequestManagerImpl extends MirrorImpl implements EventRequestManager {

    @Positive
    static int JDWPtoJDISuspendPolicy(byte jdwpPolicy);

    @Positive
    static byte JDItoJDWPSuspendPolicy(int jdiPolicy);

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    private abstract class EventRequestImpl extends MirrorImpl implements EventRequest {

    @Positive
        public boolean equals(Object obj);

    @Positive
        public int hashCode();

    @Positive
        abstract int eventCmd();

    @Positive
        InvalidRequestStateException invalidState();

    @Positive
        String state();

    @Positive
        List requestList();

    @Positive
        void delete();

    @Positive
        public boolean isEnabled();

    @Positive
        public void enable();

    @Positive
        public void disable();

    @Positive
        public synchronized void setEnabled(boolean val);

    @Positive
        public synchronized void addCountFilter(int count);

    @Positive
        public void setSuspendPolicy(int policy);

    @Positive
        public int suspendPolicy();

    @Positive
        synchronized void set();

    @Positive
        synchronized void clear();

    @Positive
        public final Object getProperty(Object key);

    @Positive
        public final void putProperty(Object key, Object value);
    @Positive
    }

    @Positive
    abstract class ThreadVisibleEventRequestImpl extends EventRequestImpl {

    @Positive
        public synchronized void addThreadFilter(ThreadReference thread);
    @Positive
    }

    @Positive
    abstract class ClassVisibleEventRequestImpl extends ThreadVisibleEventRequestImpl {

    @Positive
        public synchronized void addClassFilter(ReferenceType clazz);

    @Positive
        public synchronized void addClassFilter(String classPattern);

    @Positive
        public synchronized void addClassExclusionFilter(String classPattern);

    @Positive
        public synchronized void addInstanceFilter(ObjectReference instance);
    @Positive
    }

    @Positive
    class BreakpointRequestImpl extends ClassVisibleEventRequestImpl implements BreakpointRequest {

    @Positive
        public Location location();

    @Positive
        int eventCmd();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    class ClassPrepareRequestImpl extends ClassVisibleEventRequestImpl implements ClassPrepareRequest {

    @Positive
        int eventCmd();

    @Positive
        public synchronized void addSourceNameFilter(String sourceNamePattern);

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    class ClassUnloadRequestImpl extends ClassVisibleEventRequestImpl implements ClassUnloadRequest {

    @Positive
        int eventCmd();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    class ExceptionRequestImpl extends ClassVisibleEventRequestImpl implements ExceptionRequest {

    @Positive
        public ReferenceType exception();

    @Positive
        public boolean notifyCaught();

    @Positive
        public boolean notifyUncaught();

    @Positive
        int eventCmd();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    class MethodEntryRequestImpl extends ClassVisibleEventRequestImpl implements MethodEntryRequest {

    @Positive
        int eventCmd();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    class MethodExitRequestImpl extends ClassVisibleEventRequestImpl implements MethodExitRequest {

    @Positive
        int eventCmd();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    class MonitorContendedEnterRequestImpl extends ClassVisibleEventRequestImpl implements MonitorContendedEnterRequest {

    @Positive
        int eventCmd();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    class MonitorContendedEnteredRequestImpl extends ClassVisibleEventRequestImpl implements MonitorContendedEnteredRequest {

    @Positive
        int eventCmd();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    class MonitorWaitRequestImpl extends ClassVisibleEventRequestImpl implements MonitorWaitRequest {

    @Positive
        int eventCmd();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    class MonitorWaitedRequestImpl extends ClassVisibleEventRequestImpl implements MonitorWaitedRequest {

    @Positive
        int eventCmd();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    class StepRequestImpl extends ClassVisibleEventRequestImpl implements StepRequest {

    @Positive
        public int depth();

    @Positive
        public int size();

    @Positive
        public ThreadReference thread();

    @Positive
        int eventCmd();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    class ThreadDeathRequestImpl extends ThreadVisibleEventRequestImpl implements ThreadDeathRequest {

    @Positive
        int eventCmd();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    class ThreadStartRequestImpl extends ThreadVisibleEventRequestImpl implements ThreadStartRequest {

    @Positive
        int eventCmd();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    abstract class WatchpointRequestImpl extends ClassVisibleEventRequestImpl implements WatchpointRequest {

    @Positive
        public Field field();
    @Positive
    }

    @Positive
    class AccessWatchpointRequestImpl extends WatchpointRequestImpl implements AccessWatchpointRequest {

    @Positive
        int eventCmd();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    class ModificationWatchpointRequestImpl extends WatchpointRequestImpl implements ModificationWatchpointRequest {

    @Positive
        int eventCmd();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    class VMDeathRequestImpl extends EventRequestImpl implements VMDeathRequest {

    @Positive
        int eventCmd();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public ClassPrepareRequest createClassPrepareRequest();

    @Positive
    public ClassUnloadRequest createClassUnloadRequest();

    @Positive
    public ExceptionRequest createExceptionRequest(ReferenceType refType, boolean notifyCaught, boolean notifyUncaught);

    @Positive
    public StepRequest createStepRequest(ThreadReference thread, int size, int depth);

    @Positive
    public ThreadDeathRequest createThreadDeathRequest();

    @Positive
    public ThreadStartRequest createThreadStartRequest();

    @Positive
    public MethodEntryRequest createMethodEntryRequest();

    @Positive
    public MethodExitRequest createMethodExitRequest();

    @Positive
    public MonitorContendedEnterRequest createMonitorContendedEnterRequest();

    @Positive
    public MonitorContendedEnteredRequest createMonitorContendedEnteredRequest();

    @Positive
    public MonitorWaitRequest createMonitorWaitRequest();

    @Positive
    public MonitorWaitedRequest createMonitorWaitedRequest();

    @Positive
    public BreakpointRequest createBreakpointRequest(Location location);

    @Positive
    public AccessWatchpointRequest createAccessWatchpointRequest(Field field);

    @Positive
    public ModificationWatchpointRequest createModificationWatchpointRequest(Field field);

    @Positive
    public VMDeathRequest createVMDeathRequest();

    @Positive
    public void deleteEventRequest(EventRequest eventRequest);

    @Positive
    public void deleteEventRequests(List<? extends EventRequest> eventRequests);

    @Positive
    public void deleteAllBreakpoints();

    @Positive
    public List<StepRequest> stepRequests();

    @Positive
    public List<ClassPrepareRequest> classPrepareRequests();

    @Positive
    public List<ClassUnloadRequest> classUnloadRequests();

    @Positive
    public List<ThreadStartRequest> threadStartRequests();

    @Positive
    public List<ThreadDeathRequest> threadDeathRequests();

    @Positive
    public List<ExceptionRequest> exceptionRequests();

    @Positive
    public List<BreakpointRequest> breakpointRequests();

    @Positive
    public List<AccessWatchpointRequest> accessWatchpointRequests();

    @Positive
    public List<ModificationWatchpointRequest> modificationWatchpointRequests();

    @Positive
    public List<MethodEntryRequest> methodEntryRequests();

    @Positive
    public List<MethodExitRequest> methodExitRequests();

    @Positive
    public List<MonitorContendedEnterRequest> monitorContendedEnterRequests();

    @Positive
    public List<MonitorContendedEnteredRequest> monitorContendedEnteredRequests();

    @Positive
    public List<MonitorWaitRequest> monitorWaitRequests();

    @Positive
    public List<MonitorWaitedRequest> monitorWaitedRequests();

    @Positive
    public List<VMDeathRequest> vmDeathRequests();

    @Positive
    List<? extends EventRequest> unmodifiableRequestList(int eventCmd);

    @Positive
    EventRequest request(int eventCmd, int requestId);
    @Positive
}
