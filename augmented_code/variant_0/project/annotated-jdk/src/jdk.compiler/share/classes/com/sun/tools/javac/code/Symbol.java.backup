/*
 * Copyright (c) 1999, 2021, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.javac.code;

import org.checkerframework.checker.signature.qual.BinaryName;
import org.checkerframework.checker.signature.qual.CanonicalName;
import org.checkerframework.checker.interning.qual.InternedDistinct;
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import java.lang.annotation.Annotation;
import java.lang.annotation.Inherited;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.function.Supplier;
import java.util.function.Predicate;
import javax.lang.model.element.Element;
import javax.lang.model.element.ElementKind;
import javax.lang.model.element.ElementVisitor;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.ModuleElement;
import javax.lang.model.element.NestingKind;
import javax.lang.model.element.PackageElement;
import javax.lang.model.element.RecordComponentElement;
import javax.lang.model.element.TypeElement;
import javax.lang.model.element.TypeParameterElement;
import javax.lang.model.element.VariableElement;
import javax.tools.JavaFileManager;
import javax.tools.JavaFileObject;
import com.sun.tools.javac.code.Kinds.Kind;
import com.sun.tools.javac.comp.Annotate.AnnotationTypeMetadata;
import com.sun.tools.javac.code.Type.*;
import com.sun.tools.javac.comp.Attr;
import com.sun.tools.javac.comp.AttrContext;
import com.sun.tools.javac.comp.Env;
import com.sun.tools.javac.jvm.*;
import com.sun.tools.javac.jvm.PoolConstant;
import com.sun.tools.javac.tree.JCTree;
import com.sun.tools.javac.tree.JCTree.JCAnnotation;
import com.sun.tools.javac.tree.JCTree.JCFieldAccess;
import com.sun.tools.javac.tree.JCTree.JCVariableDecl;
import com.sun.tools.javac.tree.JCTree.Tag;
import com.sun.tools.javac.util.*;
import com.sun.tools.javac.util.DefinedBy.Api;
import com.sun.tools.javac.util.List;
import com.sun.tools.javac.util.Name;
import static com.sun.tools.javac.code.Flags.*;
import static com.sun.tools.javac.code.Kinds.*;
import static com.sun.tools.javac.code.Kinds.Kind.*;
import static com.sun.tools.javac.code.Scope.LookupKind.NON_RECURSIVE;
import com.sun.tools.javac.code.Scope.WriteableScope;
import static com.sun.tools.javac.code.TypeTag.CLASS;
import static com.sun.tools.javac.code.TypeTag.FORALL;
import static com.sun.tools.javac.code.TypeTag.TYPEVAR;
import static com.sun.tools.javac.jvm.ByteCodes.iadd;
import static com.sun.tools.javac.jvm.ByteCodes.ishll;
import static com.sun.tools.javac.jvm.ByteCodes.lushrl;
import static com.sun.tools.javac.jvm.ByteCodes.lxor;
import static com.sun.tools.javac.jvm.ByteCodes.string_add;

public abstract class Symbol extends AnnoConstruct implements PoolConstant, Element {

    public Kind kind;

    public long flags_field;

    public long flags();

    public Name name;

    public Type type;

    public Symbol owner;

    public Completer completer;

    public Type erasure_field;

    protected SymbolMetadata metadata;

    public List<Attribute.Compound> getRawAttributes();

    public List<Attribute.TypeCompound> getRawTypeAttributes();

    public Attribute.Compound attribute(Symbol anno);

    public boolean annotationsPendingCompletion();

    public void appendAttributes(List<Attribute.Compound> l);

    public void appendClassInitTypeAttributes(List<Attribute.TypeCompound> l);

    public void appendInitTypeAttributes(List<Attribute.TypeCompound> l);

    public void appendUniqueTypeAttributes(List<Attribute.TypeCompound> l);

    public List<Attribute.TypeCompound> getClassInitTypeAttributes();

    public List<Attribute.TypeCompound> getInitTypeAttributes();

    public void setInitTypeAttributes(List<Attribute.TypeCompound> l);

    public void setClassInitTypeAttributes(List<Attribute.TypeCompound> l);

    public List<Attribute.Compound> getDeclarationAttributes();

    public boolean hasAnnotations();

    public boolean hasTypeAnnotations();

    public boolean isCompleted();

    public void prependAttributes(List<Attribute.Compound> l);

    public void resetAnnotations();

    public void setAttributes(Symbol other);

    public void setDeclarationAttributes(List<Attribute.Compound> a);

    public void setTypeAttributes(List<Attribute.TypeCompound> a);

    public SymbolMetadata getMetadata();

    public Symbol(Kind kind, long flags, Name name, Type type, Symbol owner) {
    }

    @Override
    public int poolTag();

    public Symbol clone(Symbol newOwner);

    public <R, P> R accept(Symbol.Visitor<R, P> v, P p);

    @CanonicalName
    public String toString();

    public Symbol location();

    public Symbol location(Type site, Types types);

    public Symbol baseSymbol();

    public Type erasure(Types types);

    public Type externalType(Types types);

    public boolean isDeprecated();

    public boolean hasDeprecatedAnnotation();

    public boolean isDeprecatedForRemoval();

    public boolean isPreviewApi();

    public boolean isDeprecatableViaAnnotation();

    public boolean isStatic();

    public boolean isInterface();

    public boolean isAbstract();

    public boolean isPrivate();

    public boolean isPublic();

    public boolean isEnum();

    public boolean isSealed();

    public boolean isNonSealed();

    public boolean isFinal();

    public boolean isDirectlyOrIndirectlyLocal();

    public boolean isAnonymous();

    public boolean isConstructor();

    public boolean isDynamic();

    @CanonicalName
    public Name getQualifiedName();

    @BinaryName
    public Name flatName();

    public WriteableScope members();

    public boolean isInner();

    public boolean hasOuterInstance();

    public ClassSymbol enclClass();

    public ClassSymbol outermostClass();

    public PackageSymbol packge();

    public boolean isSubClass(Symbol base, Types types);

    public boolean isMemberOf(TypeSymbol clazz, Types types);

    public boolean isEnclosedBy(ClassSymbol clazz);

    public final boolean isAccessibleIn(Symbol clazz, Types types);

    public boolean isInheritedIn(Symbol clazz, Types types);

    public Symbol asMemberOf(Type site, Types types);

    public boolean overrides(Symbol _other, TypeSymbol origin, Types types, boolean checkResult);

    public void complete() throws CompletionFailure;

    public void apiComplete() throws CompletionFailure;

    public boolean exists();

    @DefinedBy(Api.LANGUAGE_MODEL)
    public Type asType();

    @DefinedBy(Api.LANGUAGE_MODEL)
    public Symbol getEnclosingElement();

    @DefinedBy(Api.LANGUAGE_MODEL)
    public ElementKind getKind();

    @DefinedBy(Api.LANGUAGE_MODEL)
    public Set<Modifier> getModifiers();

    @DefinedBy(Api.LANGUAGE_MODEL)
    public Name getSimpleName();

    @Override
    @DefinedBy(Api.LANGUAGE_MODEL)
    public List<Attribute.Compound> getAnnotationMirrors();

    @DefinedBy(Api.LANGUAGE_MODEL)
    public java.util.List<Symbol> getEnclosedElements();

    public List<TypeVariableSymbol> getTypeParameters();

    public static class DelegatedSymbol<T extends Symbol> extends Symbol {

        protected T other;

        public DelegatedSymbol(T other) {
        }

        public String toString();

        public Symbol location();

        public Symbol location(Type site, Types types);

        public Symbol baseSymbol();

        public Type erasure(Types types);

        public Type externalType(Types types);

        public boolean isDirectlyOrIndirectlyLocal();

        public boolean isConstructor();

        @CanonicalName
        public Name getQualifiedName();

        @BinaryName
        public Name flatName();

        public WriteableScope members();

        public boolean isInner();

        public boolean hasOuterInstance();

        public ClassSymbol enclClass();

        public ClassSymbol outermostClass();

        public PackageSymbol packge();

        public boolean isSubClass(Symbol base, Types types);

        public boolean isMemberOf(TypeSymbol clazz, Types types);

        public boolean isEnclosedBy(ClassSymbol clazz);

        public boolean isInheritedIn(Symbol clazz, Types types);

        public Symbol asMemberOf(Type site, Types types);

        public void complete() throws CompletionFailure;

        @DefinedBy(Api.LANGUAGE_MODEL)
        public <R, P> R accept(ElementVisitor<R, P> v, P p);

        public <R, P> R accept(Symbol.Visitor<R, P> v, P p);

        public T getUnderlyingSymbol();
    }

    public static abstract class TypeSymbol extends Symbol {

        public TypeSymbol(Kind kind, long flags, Name name, Type type, Symbol owner) {
        }

        @CanonicalName
        public static Name formFullName(Name name, Symbol owner);

        @BinaryName
        public static Name formFlatName(Name name, Symbol owner);

        public final boolean precedes(TypeSymbol that, Types types);

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public List<Symbol> getEnclosedElements();

        public AnnotationTypeMetadata getAnnotationTypeMetadata();

        public boolean isAnnotationType();

        @Override
        public <R, P> R accept(Symbol.Visitor<R, P> v, P p);
    }

    public static class TypeVariableSymbol extends TypeSymbol implements TypeParameterElement {

        public TypeVariableSymbol(long flags, Name name, Type type, Symbol owner) {
        }

        @DefinedBy(Api.LANGUAGE_MODEL)
        public ElementKind getKind();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public Symbol getGenericElement();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public List<Type> getBounds();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public List<Attribute.Compound> getAnnotationMirrors();

        @Override
        public <A extends Annotation> Attribute.Compound getAttribute(Class<A> annoType);

        boolean isCurrentSymbolsAnnotation(Attribute.TypeCompound anno, int index);

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public <R, P> R accept(ElementVisitor<R, P> v, P p);
    }

    public static class ModuleSymbol extends TypeSymbol implements ModuleElement {

        public Name version;

        public JavaFileManager.Location sourceLocation;

        public JavaFileManager.Location classLocation;

        public JavaFileManager.Location patchLocation;

        public JavaFileManager.Location patchOutputLocation;

        public List<com.sun.tools.javac.code.Directive> directives;

        public List<com.sun.tools.javac.code.Directive.RequiresDirective> requires;

        public List<com.sun.tools.javac.code.Directive.ExportsDirective> exports;

        public List<com.sun.tools.javac.code.Directive.OpensDirective> opens;

        public List<com.sun.tools.javac.code.Directive.ProvidesDirective> provides;

        public List<com.sun.tools.javac.code.Directive.UsesDirective> uses;

        public ClassSymbol module_info;

        public PackageSymbol unnamedPackage;

        public Map<Name, PackageSymbol> visiblePackages;

        public Set<ModuleSymbol> readModules;

        public List<Symbol> enclosedPackages;

        public Completer usesProvidesCompleter;

        public final Set<ModuleFlags> flags;

        public final Set<ModuleResolutionFlags> resolutionFlags;

        public static ModuleSymbol create(Name name, Name module_info);

        public ModuleSymbol(Name name, Symbol owner) {
        }

        @Override
        public int poolTag();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public Name getSimpleName();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public boolean isOpen();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public boolean isUnnamed();

        @Override
        public boolean isDeprecated();

        public boolean isNoModule();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public ElementKind getKind();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public java.util.List<Directive> getDirectives();

        public void completeUsesProvides();

        @Override
        public ClassSymbol outermostClass();

        @Override
        public String toString();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public <R, P> R accept(ElementVisitor<R, P> v, P p);

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public List<Symbol> getEnclosedElements();

        public void reset();
    }

    public enum ModuleFlags {

        OPEN(0x0020), SYNTHETIC(0x1000), MANDATED(0x8000);

        public static int value(Set<ModuleFlags> s);

        public final int value;
    }

    public enum ModuleResolutionFlags {

        DO_NOT_RESOLVE_BY_DEFAULT(0x0001), WARN_DEPRECATED(0x0002), WARN_DEPRECATED_REMOVAL(0x0004), WARN_INCUBATING(0x0008);

        public static int value(Set<ModuleResolutionFlags> s);

        public final int value;
    }

    public static class PackageSymbol extends TypeSymbol implements PackageElement {

        public WriteableScope members_field;

        public Name fullname;

        public ClassSymbol package_info;

        public ModuleSymbol modle;

        public JavaFileObject sourcefile;

        public PackageSymbol(Name name, Type type, Symbol owner) {
        }

        public PackageSymbol(Name name, Symbol owner) {
        }

        public String toString();

        @DefinedBy(Api.LANGUAGE_MODEL)
        @CanonicalName
        public Name getQualifiedName();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public boolean isUnnamed();

        public WriteableScope members();

        @Override
        public int poolTag();

        public long flags();

        @Override
        public List<Attribute.Compound> getRawAttributes();

        public boolean exists();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public ElementKind getKind();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public Symbol getEnclosingElement();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public <R, P> R accept(ElementVisitor<R, P> v, P p);

        public <R, P> R accept(Symbol.Visitor<R, P> v, P p);

        public void reset();
    }

    public static class RootPackageSymbol extends PackageSymbol {

        public final MissingInfoHandler missingInfoHandler;

        public final boolean allowPrivateInvokeVirtual;

        public RootPackageSymbol(Name name, Symbol owner, MissingInfoHandler missingInfoHandler, boolean allowPrivateInvokeVirtual) {
        }
    }

    public static class ClassSymbol extends TypeSymbol implements TypeElement {

        public WriteableScope members_field;

        public Name fullname;

        @BinaryName
        public Name flatname;

        public JavaFileObject sourcefile;

        public JavaFileObject classfile;

        public List<ClassSymbol> trans_local;

        public List<Symbol> permitted;

        public boolean isPermittedExplicit;

        public ClassSymbol(long flags, Name name, Type type, Symbol owner) {
        }

        public ClassSymbol(long flags, Name name, Symbol owner) {
        }

        public String toString();

        public long flags();

        public WriteableScope members();

        @Override
        public List<Attribute.Compound> getRawAttributes();

        @Override
        public List<Attribute.TypeCompound> getRawTypeAttributes();

        public Type erasure(Types types);

        public String className();

        @DefinedBy(Api.LANGUAGE_MODEL)
        @CanonicalName
        public Name getQualifiedName();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public List<Symbol> getEnclosedElements();

        @BinaryName
        public Name flatName();

        public boolean isSubClass(Symbol base, Types types);

        public void complete() throws CompletionFailure;

        @DefinedBy(Api.LANGUAGE_MODEL)
        public List<Type> getInterfaces();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public Type getSuperclass();

        @Override
        protected <A extends Annotation> A[] getInheritedAnnotations(Class<A> annoType);

        @DefinedBy(Api.LANGUAGE_MODEL)
        public ElementKind getKind();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public Set<Modifier> getModifiers();

        public RecordComponent getRecordComponent(VarSymbol field);

        public RecordComponent getRecordComponent(JCVariableDecl var, boolean addIfMissing, List<JCAnnotation> annotations);

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public List<? extends RecordComponent> getRecordComponents();

        public void setRecordComponents(List<RecordComponent> recordComponents);

        @DefinedBy(Api.LANGUAGE_MODEL)
        public NestingKind getNestingKind();

        @Override
        protected <A extends Annotation> Attribute.Compound getAttribute(final Class<A> annoType);

        @DefinedBy(Api.LANGUAGE_MODEL)
        public <R, P> R accept(ElementVisitor<R, P> v, P p);

        public <R, P> R accept(Symbol.Visitor<R, P> v, P p);

        public void markAbstractIfNeeded(Types types);

        public void reset();

        public void clearAnnotationMetadata();

        @Override
        public AnnotationTypeMetadata getAnnotationTypeMetadata();

        @Override
        public boolean isAnnotationType();

        public void setAnnotationTypeMetadata(AnnotationTypeMetadata a);

        public boolean isRecord();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public List<Type> getPermittedSubclasses();
    }

    public static class VarSymbol extends Symbol implements VariableElement {

        public int pos;

        public int adr;

        public VarSymbol(long flags, Name name, Type type, Symbol owner) {
        }

        @Override
        public int poolTag();

        public MethodHandleSymbol asMethodHandle(boolean getter);

        public VarSymbol clone(Symbol newOwner);

        public String toString();

        public Symbol asMemberOf(Type site, Types types);

        @DefinedBy(Api.LANGUAGE_MODEL)
        public ElementKind getKind();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public <R, P> R accept(ElementVisitor<R, P> v, P p);

        @DefinedBy(Api.LANGUAGE_MODEL)
        public Object getConstantValue();

        public void setLazyConstValue(final Env<AttrContext> env, final Attr attr, final JCVariableDecl variable);

        public boolean isExceptionParameter();

        public boolean isResourceVariable();

        public Object getConstValue();

        public void setData(Object data);

        public <R, P> R accept(Symbol.Visitor<R, P> v, P p);
    }

    public static class RecordComponent extends VarSymbol implements RecordComponentElement {

        public MethodSymbol accessor;

        public JCTree.JCMethodDecl accessorMeth;

        public RecordComponent(Name name, Type type, Symbol owner) {
        }

        public RecordComponent(VarSymbol field, List<JCAnnotation> annotations) {
        }

        public List<JCAnnotation> getOriginalAnnos();

        public boolean isVarargs();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public ElementKind getKind();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public ExecutableElement getAccessor();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public <R, P> R accept(ElementVisitor<R, P> v, P p);
    }

    public static class ParamSymbol extends VarSymbol {

        public ParamSymbol(long flags, Name name, Type type, Symbol owner) {
        }

        @Override
        public Name getSimpleName();
    }

    public static class BindingSymbol extends VarSymbol {

        public BindingSymbol(long flags, Name name, Type type, Symbol owner) {
        }

        public boolean isAliasFor(BindingSymbol b);

        List<BindingSymbol> aliases();

        public void preserveBinding();

        public boolean isPreserved();
    }

    public static class MethodSymbol extends Symbol implements ExecutableElement {

        public Code code;

        public List<VarSymbol> extraParams;

        public List<VarSymbol> capturedLocals;

        public List<VarSymbol> params;

        public Attribute defaultValue;

        public MethodSymbol(long flags, Name name, Type type, Symbol owner) {
        }

        public MethodSymbol clone(Symbol newOwner);

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public Set<Modifier> getModifiers();

        public String toString();

        @Override
        public int poolTag();

        public boolean isHandle();

        public MethodHandleSymbol asHandle();

        public Symbol implemented(TypeSymbol c, Types types);

        public Symbol implementedIn(TypeSymbol c, Types types);

        public boolean binaryOverrides(Symbol _other, TypeSymbol origin, Types types);

        public MethodSymbol binaryImplementation(ClassSymbol origin, Types types);

        public boolean overrides(Symbol _other, TypeSymbol origin, Types types, boolean checkResult);

        public boolean overrides(Symbol _other, TypeSymbol origin, Types types, boolean checkResult, boolean requireConcreteIfInherited);

        @Override
        public boolean isInheritedIn(Symbol clazz, Types types);

        public boolean isLambdaMethod();

        public MethodSymbol originalEnclosingMethod();

        public MethodSymbol implementation(TypeSymbol origin, Types types, boolean checkResult);

        public static final Predicate<Symbol> implementation_filter;

        public MethodSymbol implementation(TypeSymbol origin, Types types, boolean checkResult, Predicate<Symbol> implFilter);

        public List<VarSymbol> params();

        public Symbol asMemberOf(Type site, Types types);

        @DefinedBy(Api.LANGUAGE_MODEL)
        public ElementKind getKind();

        public boolean isStaticOrInstanceInit();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public Attribute getDefaultValue();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public List<VarSymbol> getParameters();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public boolean isVarArgs();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public boolean isDefault();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public <R, P> R accept(ElementVisitor<R, P> v, P p);

        public <R, P> R accept(Symbol.Visitor<R, P> v, P p);

        @DefinedBy(Api.LANGUAGE_MODEL)
        public Type getReceiverType();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public Type getReturnType();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public List<Type> getThrownTypes();
    }

    public static class DynamicMethodSymbol extends MethodSymbol implements Dynamic {

        public LoadableConstant[] staticArgs;

        public MethodHandleSymbol bsm;

        public DynamicMethodSymbol(Name name, Symbol owner, MethodHandleSymbol bsm, Type type, LoadableConstant[] staticArgs) {
        }

        @Override
        public boolean isDynamic();

        @Override
        public LoadableConstant[] staticArgs();

        @Override
        public MethodHandleSymbol bootstrapMethod();

        @Override
        public int poolTag();

        @Override
        public Type dynamicType();
    }

    public static class DynamicVarSymbol extends VarSymbol implements Dynamic, LoadableConstant {

        public LoadableConstant[] staticArgs;

        public MethodHandleSymbol bsm;

        public DynamicVarSymbol(Name name, Symbol owner, MethodHandleSymbol bsm, Type type, LoadableConstant[] staticArgs) {
        }

        @Override
        public boolean isDynamic();

        @Override
        public PoolConstant dynamicType();

        @Override
        public LoadableConstant[] staticArgs();

        @Override
        public LoadableConstant bootstrapMethod();

        @Override
        public int poolTag();
    }

    public static class MethodHandleSymbol extends MethodSymbol implements LoadableConstant {

        public MethodHandleSymbol(Symbol msym) {
        }

        public MethodHandleSymbol(Symbol msym, boolean getter) {
        }

        public int referenceKind();

        @Override
        public int poolTag();

        @Override
        public Object poolKey(Types types);

        @Override
        public MethodHandleSymbol asHandle();

        @Override
        public Symbol baseSymbol();

        @Override
        public boolean isHandle();
    }

    public static class OperatorSymbol extends MethodSymbol {

        public int opcode;

        public OperatorSymbol(Name name, Type type, int opcode, Symbol owner) {
        }

        @Override
        public <R, P> R accept(Symbol.Visitor<R, P> v, P p);

        public int getAccessCode(Tag tag);

        public enum AccessCode {

            UNKNOWN(-1, Tag.NO_TAG),
            DEREF(0, Tag.NO_TAG),
            ASSIGN(2, Tag.ASSIGN),
            PREINC(4, Tag.PREINC),
            PREDEC(6, Tag.PREDEC),
            POSTINC(8, Tag.POSTINC),
            POSTDEC(10, Tag.POSTDEC),
            FIRSTASGOP(12, Tag.NO_TAG);

            public final int code;

            public final Tag tag;

            public static final int numberOfAccessCodes;

            public static AccessCode getFromCode(int code);

            static int from(Tag tag, int opcode);
        }
    }

    @UsesObjectEquals
    public static interface Completer {

        @InternedDistinct
        public static final Completer NULL_COMPLETER;

        void complete(Symbol sym) throws CompletionFailure;

        default boolean isTerminal();
    }

    public static class CompletionFailure extends RuntimeException {

        public final transient DeferredCompletionFailureHandler dcfh;

        public transient Symbol sym;

        public CompletionFailure(Symbol sym, Supplier<JCDiagnostic> diagSupplier, DeferredCompletionFailureHandler dcfh) {
        }

        public JCDiagnostic getDiagnostic();

        @Override
        public String getMessage();

        public JCDiagnostic getDetailValue();

        @Override
        public CompletionFailure initCause(Throwable cause);

        public void resetDiagnostic(Supplier<JCDiagnostic> diagSupplier);
    }

    public interface Visitor<R, P> {

        R visitClassSymbol(ClassSymbol s, P arg);

        R visitMethodSymbol(MethodSymbol s, P arg);

        R visitPackageSymbol(PackageSymbol s, P arg);

        R visitOperatorSymbol(OperatorSymbol s, P arg);

        R visitVarSymbol(VarSymbol s, P arg);

        R visitTypeSymbol(TypeSymbol s, P arg);

        R visitSymbol(Symbol s, P arg);
    }
}
