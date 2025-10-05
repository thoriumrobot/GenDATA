/*
    @Positive
 * Copyright (c) 2006, 2021, Oracle and/or its affiliates. All rights reserved.
    @Positive
 */
    @Positive
package com.sun.org.apache.xerces.internal.impl.xs;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import com.sun.org.apache.xerces.internal.impl.Constants;
    @Positive
import com.sun.org.apache.xerces.internal.impl.RevalidationHandler;
    @Positive
import com.sun.org.apache.xerces.internal.impl.XMLEntityManager;
    @Positive
import com.sun.org.apache.xerces.internal.impl.XMLErrorReporter;
    @Positive
import com.sun.org.apache.xerces.internal.impl.dv.DatatypeException;
    @Positive
import com.sun.org.apache.xerces.internal.impl.dv.InvalidDatatypeValueException;
    @Positive
import com.sun.org.apache.xerces.internal.impl.dv.ValidatedInfo;
    @Positive
import com.sun.org.apache.xerces.internal.impl.dv.XSSimpleType;
    @Positive
import com.sun.org.apache.xerces.internal.impl.dv.xs.XSSimpleTypeDecl;
    @Positive
import com.sun.org.apache.xerces.internal.impl.validation.ConfigurableValidationState;
    @Positive
import com.sun.org.apache.xerces.internal.impl.validation.ValidationManager;
    @Positive
import com.sun.org.apache.xerces.internal.impl.validation.ValidationState;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xs.identity.Field;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xs.identity.FieldActivator;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xs.identity.IdentityConstraint;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xs.identity.KeyRef;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xs.identity.Selector;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xs.identity.UniqueOrKey;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xs.identity.ValueStore;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xs.identity.XPathMatcher;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xs.models.CMBuilder;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xs.models.CMNodeFactory;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xs.models.XSCMValidator;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xs.util.XS10TypeHelper;
    @Positive
import com.sun.org.apache.xerces.internal.parsers.XMLParser;
    @Positive
import com.sun.org.apache.xerces.internal.util.AugmentationsImpl;
    @Positive
import com.sun.org.apache.xerces.internal.util.IntStack;
    @Positive
import com.sun.org.apache.xerces.internal.util.SymbolTable;
    @Positive
import com.sun.org.apache.xerces.internal.util.URI.MalformedURIException;
    @Positive
import com.sun.org.apache.xerces.internal.util.XMLAttributesImpl;
    @Positive
import com.sun.org.apache.xerces.internal.util.XMLChar;
    @Positive
import com.sun.org.apache.xerces.internal.util.XMLSymbols;
    @Positive
import com.sun.org.apache.xerces.internal.xni.Augmentations;
    @Positive
import com.sun.org.apache.xerces.internal.xni.NamespaceContext;
    @Positive
import com.sun.org.apache.xerces.internal.xni.QName;
    @Positive
import com.sun.org.apache.xerces.internal.xni.XMLAttributes;
    @Positive
import com.sun.org.apache.xerces.internal.xni.XMLDocumentHandler;
    @Positive
import com.sun.org.apache.xerces.internal.xni.XMLLocator;
    @Positive
import com.sun.org.apache.xerces.internal.xni.XMLResourceIdentifier;
    @Positive
import com.sun.org.apache.xerces.internal.xni.XMLString;
    @Positive
import com.sun.org.apache.xerces.internal.xni.XNIException;
    @Positive
import com.sun.org.apache.xerces.internal.xni.grammars.XMLGrammarDescription;
    @Positive
import com.sun.org.apache.xerces.internal.xni.grammars.XMLGrammarPool;
    @Positive
import com.sun.org.apache.xerces.internal.xni.parser.XMLComponent;
    @Positive
import com.sun.org.apache.xerces.internal.xni.parser.XMLComponentManager;
    @Positive
import com.sun.org.apache.xerces.internal.xni.parser.XMLConfigurationException;
    @Positive
import com.sun.org.apache.xerces.internal.xni.parser.XMLDocumentFilter;
    @Positive
import com.sun.org.apache.xerces.internal.xni.parser.XMLDocumentSource;
    @Positive
import com.sun.org.apache.xerces.internal.xni.parser.XMLEntityResolver;
    @Positive
import com.sun.org.apache.xerces.internal.xni.parser.XMLInputSource;
    @Positive
import com.sun.org.apache.xerces.internal.xs.AttributePSVI;
    @Positive
import com.sun.org.apache.xerces.internal.xs.ElementPSVI;
    @Positive
import com.sun.org.apache.xerces.internal.xs.ShortList;
    @Positive
import com.sun.org.apache.xerces.internal.xs.StringList;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSConstants;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSObjectList;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSTypeDefinition;
    @Positive
import java.io.IOException;
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
import java.util.Stack;
    @Positive
import java.util.Vector;
    @Positive
import javax.xml.XMLConstants;
    @Positive
import jdk.xml.internal.JdkConstants;
    @Positive
import jdk.xml.internal.JdkXmlUtils;

    @Positive
public class XMLSchemaValidator implements XMLComponent, XMLDocumentFilter, FieldActivator, RevalidationHandler, XSElementDeclHelper {

    @Positive
    protected static final String VALIDATION;

    @Positive
    protected static final String SCHEMA_VALIDATION;

    @Positive
    protected static final String SCHEMA_FULL_CHECKING;

    @Positive
    protected static final String DYNAMIC_VALIDATION;

    @Positive
    protected static final String NORMALIZE_DATA;

    @Positive
    protected static final String SCHEMA_ELEMENT_DEFAULT;

    @Positive
    protected static final String SCHEMA_AUGMENT_PSVI;

    @Positive
    protected static final String ALLOW_JAVA_ENCODINGS;

    @Positive
    protected static final String STANDARD_URI_CONFORMANT_FEATURE;

    @Positive
    protected static final String GENERATE_SYNTHETIC_ANNOTATIONS;

    @Positive
    protected static final String VALIDATE_ANNOTATIONS;

    @Positive
    protected static final String HONOUR_ALL_SCHEMALOCATIONS;

    @Positive
    protected static final String USE_GRAMMAR_POOL_ONLY;

    @Positive
    protected static final String CONTINUE_AFTER_FATAL_ERROR;

    @Positive
    protected static final String PARSER_SETTINGS;

    @Positive
    protected static final String NAMESPACE_GROWTH;

    @Positive
    protected static final String TOLERATE_DUPLICATES;

    @Positive
    protected static final String IGNORE_XSI_TYPE;

    @Positive
    protected static final String ID_IDREF_CHECKING;

    @Positive
    protected static final String UNPARSED_ENTITY_CHECKING;

    @Positive
    protected static final String IDENTITY_CONSTRAINT_CHECKING;

    @Positive
    protected static final String REPORT_WHITESPACE;

    @Positive
    public static final String SYMBOL_TABLE;

    @Positive
    public static final String ERROR_REPORTER;

    @Positive
    public static final String ENTITY_RESOLVER;

    @Positive
    public static final String XMLGRAMMAR_POOL;

    @Positive
    protected static final String VALIDATION_MANAGER;

    @Positive
    protected static final String ENTITY_MANAGER;

    @Positive
    protected static final String SCHEMA_LOCATION;

    @Positive
    protected static final String SCHEMA_NONS_LOCATION;

    @Positive
    protected static final String JAXP_SCHEMA_SOURCE;

    @Positive
    protected static final String JAXP_SCHEMA_LANGUAGE;

    @Positive
    protected static final String ROOT_TYPE_DEF;

    @Positive
    protected static final String ROOT_ELEMENT_DECL;

    @Positive
    protected static final String SCHEMA_DV_FACTORY;

    @Positive
    protected static final String OVERRIDE_PARSER;

    @Positive
    protected static final String USE_CATALOG;

    @Positive
    protected static final int ID_CONSTRAINT_NUM;

    @Positive
    protected ElementPSVImpl fCurrentPSVI;

    @Positive
    protected final AugmentationsImpl fAugmentations;

    @Positive
    protected XMLString fDefaultValue;

    @Positive
    protected boolean fDynamicValidation;

    @Positive
    protected boolean fSchemaDynamicValidation;

    @Positive
    protected boolean fDoValidation;

    @Positive
    protected boolean fFullChecking;

    @Positive
    protected boolean fNormalizeData;

    @Positive
    protected boolean fSchemaElementDefault;

    @Positive
    protected boolean fAugPSVI;

    @Positive
    protected boolean fIdConstraint;

    @Positive
    protected boolean fUseGrammarPoolOnly;

    @Positive
    protected boolean fNamespaceGrowth;

    @Positive
    protected boolean fEntityRef;

    @Positive
    protected boolean fInCDATA;

    @Positive
    protected boolean fSawOnlyWhitespaceInElementContent;

    @Positive
    protected SymbolTable fSymbolTable;

    @Positive
    protected final class XSIErrorReporter {

    @Positive
        public void reset(XMLErrorReporter errorReporter);

    @Positive
        public void pushContext();

    @Positive
        public String[] popContext();

    @Positive
        public String[] mergeContext();

    @Positive
        public void reportError(String domain, String key, Object[] arguments, short severity) throws XNIException;

    @Positive
        public void reportError(XMLLocator location, String domain, String key, Object[] arguments, short severity) throws XNIException;
    @Positive
    }

    @Positive
    protected final XSIErrorReporter fXSIErrorReporter;

    @Positive
    protected XMLEntityResolver fEntityResolver;

    @Positive
    protected ValidationManager fValidationManager;

    @Positive
    protected ConfigurableValidationState fValidationState;

    @Positive
    protected XMLGrammarPool fGrammarPool;

    @Positive
    protected String fExternalSchemas;

    @Positive
    protected String fExternalNoNamespaceSchema;

    @Positive
    protected Object fJaxpSchemaSource;

    @Positive
    protected final XSDDescription fXSDDescription;

    @Positive
    protected final Map<String, XMLSchemaLoader.LocationArray> fLocationPairs;

    @Positive
    protected XMLDocumentHandler fDocumentHandler;

    @Positive
    protected XMLDocumentSource fDocumentSource;

    @Positive
    public String[] getRecognizedFeatures();

    @Positive
    public void setFeature(String featureId, boolean state) throws XMLConfigurationException;

    @Positive
    public String[] getRecognizedProperties();

    @Positive
    public void setProperty(String propertyId, Object value) throws XMLConfigurationException;

    @Positive
    public Boolean getFeatureDefault(String featureId);

    @Positive
    public Object getPropertyDefault(String propertyId);

    @Positive
    public void setDocumentHandler(XMLDocumentHandler documentHandler);

    @Positive
    public XMLDocumentHandler getDocumentHandler();

    @Positive
    public void setDocumentSource(XMLDocumentSource source);

    @Positive
    public XMLDocumentSource getDocumentSource();

    @Positive
    public void startDocument(XMLLocator locator, String encoding, NamespaceContext namespaceContext, Augmentations augs) throws XNIException;

    @Positive
    public void xmlDecl(String version, String encoding, String standalone, Augmentations augs) throws XNIException;

    @Positive
    public void doctypeDecl(String rootElement, String publicId, String systemId, Augmentations augs) throws XNIException;

    @Positive
    public void startElement(QName element, XMLAttributes attributes, Augmentations augs) throws XNIException;

    @Positive
    public void emptyElement(QName element, XMLAttributes attributes, Augmentations augs) throws XNIException;

    @Positive
    public void characters(XMLString text, Augmentations augs) throws XNIException;

    @Positive
    public void ignorableWhitespace(XMLString text, Augmentations augs) throws XNIException;

    @Positive
    public void endElement(QName element, Augmentations augs) throws XNIException;

    @Positive
    public void startCDATA(Augmentations augs) throws XNIException;

    @Positive
    public void endCDATA(Augmentations augs) throws XNIException;

    @Positive
    public void endDocument(Augmentations augs) throws XNIException;

    @Positive
    public boolean characterData(String data, Augmentations augs);

    @Positive
    public void elementDefault(String data);

    @Positive
    public void startGeneralEntity(String name, XMLResourceIdentifier identifier, String encoding, Augmentations augs) throws XNIException;

    @Positive
    public void textDecl(String version, String encoding, Augmentations augs) throws XNIException;

    @Positive
    public void comment(XMLString text, Augmentations augs) throws XNIException;

    @Positive
    public void processingInstruction(String target, XMLString data, Augmentations augs) throws XNIException;

    @Positive
    public void endGeneralEntity(String name, Augmentations augs) throws XNIException;

    @Positive
    protected XPathMatcherStack fMatcherStack;

    @Positive
    protected ValueStoreCache fValueStoreCache;

    @Positive
    public XMLSchemaValidator() {
    @Positive
    }

    @Positive
    public void reset(XMLComponentManager componentManager) throws XMLConfigurationException;

    @Positive
    public void startValueScopeFor(IdentityConstraint identityConstraint, int initialDepth);

    @Positive
    public XPathMatcher activateField(Field field, int initialDepth);

    @Positive
    public void endValueScopeFor(IdentityConstraint identityConstraint, int initialDepth);

    @Positive
    public XSElementDecl getGlobalElementDecl(QName element);

    @Positive
    void ensureStackCapacity();

    @Positive
    void handleStartDocument(XMLLocator locator, String encoding);

    @Positive
    void handleEndDocument();

    @Positive
    XMLString handleCharacters(XMLString text);

    @Positive
    void handleIgnorableWhitespace(XMLString text);

    @Positive
    Augmentations handleStartElement(QName element, XMLAttributes attributes, Augmentations augs);

    @Positive
    Augmentations handleEndElement(QName element, Augmentations augs);

    @Positive
    final Augmentations endElementPSVI(boolean root, SchemaGrammar[] grammars, Augmentations augs);

    @Positive
    Augmentations getEmptyAugs(Augmentations augs);

    @Positive
    void storeLocations(String sLocation, String nsLocation);

    @Positive
    SchemaGrammar findSchemaGrammar(short contextType, String namespace, QName enclosingElement, QName triggeringComponent, XMLAttributes attributes);

    @Positive
    XSTypeDefinition getAndCheckXsiType(QName element, String xsiType, XMLAttributes attributes);

    @Positive
    boolean getXsiNil(QName element, String xsiNil);

    @Positive
    void processAttributes(QName element, XMLAttributes attributes, XSAttributeGroupDecl attrGrp);

    @Positive
    void processOneAttribute(QName element, XMLAttributes attributes, int index, XSAttributeDecl currDecl, XSAttributeUseImpl currUse, AttributePSVImpl attrPSVI);

    @Positive
    void addDefaultAttributes(QName element, XMLAttributes attributes, XSAttributeGroupDecl attrGrp);

    @Positive
    void processElementContent(QName element);

    @Positive
    Object elementLocallyValidType(QName element, Object textContent);

    @Positive
    Object elementLocallyValidComplexType(QName element, Object textContent);

    @Positive
    void processRootTypeQName(final javax.xml.namespace.QName rootTypeQName);

    @Positive
    void processRootElementDeclQName(final javax.xml.namespace.QName rootElementDeclQName, final QName element);

    @Positive
    void checkElementMatchesRootElementDecl(final XSElementDecl rootElementDecl, final QName element);

    @Positive
    void reportSchemaError(String key, Object[] arguments);

    @Positive
    protected static class XPathMatcherStack {

    @Positive
        protected XPathMatcher[] fMatchers;

    @Positive
        protected int fMatchersCount;

    @Positive
        protected IntStack fContextStack;

    @Positive
        public XPathMatcherStack() {
    @Positive
        }

    @Positive
        public void clear();

    @Positive
        public int size();

    @Positive
        public int getMatcherCount();

    @Positive
        public void addMatcher(XPathMatcher matcher);

    @Positive
        public XPathMatcher getMatcherAt(int index);

    @Positive
        public void pushContext();

    @Positive
        public void popContext();
    @Positive
    }

    @Positive
    protected abstract class ValueStoreBase implements ValueStore {

    @Positive
        protected IdentityConstraint fIdentityConstraint;

    @Positive
        protected int fFieldCount;

    @Positive
        protected Field[] fFields;

    @Positive
        protected Object[] fLocalValues;

    @Positive
        protected short[] fLocalValueTypes;

    @Positive
        protected ShortList[] fLocalItemValueTypes;

    @Positive
        protected int fValuesCount;

    @Positive
        protected boolean fHasValue;

    @Positive
        public final Vector<Object> fValues;

    @Positive
        public ShortVector fValueTypes;

    @Positive
        public Vector<ShortList> fItemValueTypes;

    @Positive
        protected ValueStoreBase(IdentityConstraint identityConstraint) {
    @Positive
        }

    @Positive
        public void clear();

    @Positive
        public void append(ValueStoreBase newVal);

    @Positive
        public void startValueScope();

    @Positive
        public void endValueScope();

    @Positive
        public void endDocumentFragment();

    @Positive
        public void endDocument();

    @Positive
        public void reportError(String key, Object[] args);

    @Positive
        public void addValue(Field field, boolean mayMatch, Object actualValue, short valueType, ShortList itemValueType);

    @Positive
        @Pure
    @Positive
        public boolean contains();

    @Positive
        public int contains(ValueStoreBase vsb);

    @Positive
        protected void checkDuplicateValues();

    @Positive
        protected String toString(Object[] values);

    @Positive
        protected String toString(Vector<Object> values, int start, int length);

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    protected class UniqueValueStore extends ValueStoreBase {

    @Positive
        public UniqueValueStore(UniqueOrKey unique) {
    @Positive
        }

    @Positive
        protected void checkDuplicateValues();
    @Positive
    }

    @Positive
    protected class KeyValueStore extends ValueStoreBase {

    @Positive
        public KeyValueStore(UniqueOrKey key) {
    @Positive
        }

    @Positive
        protected void checkDuplicateValues();
    @Positive
    }

    @Positive
    protected class KeyRefValueStore extends ValueStoreBase {

    @Positive
        protected ValueStoreBase fKeyValueStore;

    @Positive
        public KeyRefValueStore(KeyRef keyRef, KeyValueStore keyValueStore) {
    @Positive
        }

    @Positive
        public void endDocumentFragment();

    @Positive
        public void endDocument();
    @Positive
    }

    @Positive
    protected class ValueStoreCache {

    @Positive
        protected final List<ValueStoreBase> fValueStores;

    @Positive
        protected final Map<LocalIDKey, ValueStoreBase> fIdentityConstraint2ValueStoreMap;

    @Positive
        protected final Stack<Map<IdentityConstraint, ValueStoreBase>> fGlobalMapStack;

    @Positive
        protected final Map<IdentityConstraint, ValueStoreBase> fGlobalIDConstraintMap;

    @Positive
        public ValueStoreCache() {
    @Positive
        }

    @Positive
        public void startDocument();

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        public void startElement();

    @Positive
        public void endElement();

    @Positive
        public void initValueStoresFor(XSElementDecl eDecl, FieldActivator activator);

    @Positive
        public ValueStoreBase getValueStoreFor(IdentityConstraint id, int initialDepth);

    @Positive
        public ValueStoreBase getGlobalValueStoreFor(IdentityConstraint id);

    @Positive
        public void transplant(IdentityConstraint id, int initialDepth);

    @Positive
        public void endDocument();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    protected static final class LocalIDKey {

    @Positive
        public IdentityConstraint fId;

    @Positive
        public int fDepth;

    @Positive
        public LocalIDKey() {
    @Positive
        }

    @Positive
        public LocalIDKey(IdentityConstraint id, int depth) {
    @Positive
        }

    @Positive
        public int hashCode();

    @Positive
        public boolean equals(Object localIDKey);
    @Positive
    }

    @Positive
    protected static final class ShortVector {

    @Positive
        public ShortVector() {
    @Positive
        }

    @Positive
        public ShortVector(int initialCapacity) {
    @Positive
        }

    @Positive
        public int length();

    @Positive
        public void add(short value);

    @Positive
        public short valueAt(int position);

    @Positive
        public void clear();

    @Positive
        @Pure
    @Positive
        public boolean contains(short value);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
